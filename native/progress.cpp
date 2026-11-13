// SPDX-License-Identifier: MIT
// Native console observer; closing its terminal never closes the host/builder.
#include <windows.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>

namespace {
std::string read(const std::filesystem::path& path) {
    // path: Atomically published UTF-8 status or append-only builder log.
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void print(HANDLE console, const std::string& value) {
    // console/value: Real CONOUT$ handle and UTF-8 text, independent of host stdio.
    std::string normalized;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\n' && (i == 0 || value[i-1] != '\r')) normalized += '\r';
        normalized += value[i];
    }
    int count = MultiByteToWideChar(CP_UTF8, 0, normalized.data(), int(normalized.size()), nullptr, 0);
    std::wstring text(count, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, normalized.data(), int(normalized.size()), text.data(), count);
    DWORD written{};
    WriteConsoleW(console, text.data(), DWORD(text.size()), &written, nullptr);
}
}

int wmain(int argc, wchar_t** argv) {
    // argv: Absolute session-status path and parent PID; no shell is executed.
    if (argc != 3) return 2;
    HANDLE console = CreateFileW(L"CONOUT$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                 nullptr, OPEN_EXISTING, 0, nullptr);
    DWORD mode{};
    if (console == INVALID_HANDLE_VALUE || !GetConsoleMode(console, &mode)) return 3;
    HANDLE parent{};
    try { parent = OpenProcess(SYNCHRONIZE, FALSE, DWORD(std::stoul(argv[2]))); }
    catch (...) { CloseHandle(console); return 2; }
    SetConsoleTitleW(L"RIFE-SuperBoost - TensorRT engine preparation");
    auto started = GetTickCount64(), last_tick = started;
    ULONGLONG deadline = 0;
    std::string previous, log_path;
    size_t offset = 0;
    print(console, "RIFE-SuperBoost: preparing GPU engines. First use can take several minutes.\n"
                   "Encoder and synthesis share this terminal.\n"
                   "Closing this terminal does not cancel preparation.\n\n");
    for (;;) {
        auto now = GetTickCount64();
        auto status = read(argv[1]);
        if (!status.empty() && status != previous) {
            previous = status;
            std::istringstream lines(status);
            std::string state, path, message;
            std::getline(lines, state); std::getline(lines, path);
            if (!state.empty() && state.back() == '\r') state.pop_back();
            if (!path.empty() && path.back() == '\r') path.pop_back();
            message.assign(std::istreambuf_iterator<char>(lines), std::istreambuf_iterator<char>());
            if (path != log_path) { log_path = path; offset = 0; }
            print(console, "\n" + message + "\n");
            deadline = state == "complete" ? now + 2000 : state == "failed" ? now + 15000 : 0;
        }
        if (!log_path.empty()) {
            auto log = read(std::filesystem::u8path(log_path));
            if (log.size() > offset) { print(console, log.substr(offset)); offset = log.size(); }
        }
        if (deadline && now >= deadline) break;
        if (now - last_tick >= 1000) {
            auto title = L"RIFE-SuperBoost - TensorRT engine preparation (" + std::to_wstring((now-started)/1000) + L" s)";
            SetConsoleTitleW(title.c_str());
            last_tick = now;
        }
        if (!deadline && (!parent || WaitForSingleObject(parent, 0) != WAIT_TIMEOUT)) {
            print(console, "The preparation host exited.\n"); deadline = now + 2000;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (parent) CloseHandle(parent);
    CloseHandle(console);
    return 0;
}
