# SPDX-License-Identifier: MIT
"""One native console per engine-preparation session, isolated from the player."""

import os
from pathlib import Path
import subprocess
import uuid


class BuildFeedback:
    """Own a lazy observer; warm cache hits create neither files nor a process."""

    def __init__(self, native, folder: Path, message: str, visible: bool, initially_active: bool = True) -> None:
        """Args:
            native: Retained backend containing the standalone progress executable.
            folder: Writable engine cache entry.
            message: Model, role, shape, precision and current preparation state.
            visible: Open an independent progress window when True.
            initially_active: False defers all files/processes until a cache miss.
        """
        self.path = folder / ("status-" + uuid.uuid4().hex + ".txt")
        self.log = ""
        self.message = message
        self.viewer = None
        self.native, self.visible, self.active = native, visible, False
        self.stage_index, self.stage_count = 0, 1
        if initially_active:
            self.start()

    def start(self) -> None:
        """Activate at most once, before waiting or beginning a cold build."""
        if self.active:
            return
        self.active = True
        self.update("waiting", "Checking cache or waiting for another builder")
        if self.visible:
            executable = self.native.folder / "rsb_progress.exe"
            if not executable.is_file():
                raise FileNotFoundError(f"Build feedback executable is missing: {executable}")
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 1  # SW_SHOWNORMAL on the normal interactive desktop.
            self.viewer = subprocess.Popen([str(executable), str(self.path), str(os.getpid())],
                                           creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)

    def set_stage(self, index: int, count: int) -> None:
        """Args:
            index: One-based encoder/synthesis stage number in this session.
            count: Total engine roles; this is not an overall time percentage.
        """
        self.stage_index, self.stage_count = index, count

    def begin(self, message: str, detail: str = "Checking cache or waiting for another builder") -> None:
        """Args:
            message: Model, role, geometry, precision and cache path for this stage.
            detail: Current preparation state, shown in the existing console.
        """
        prefix = f"[{self.stage_index}/{self.stage_count}] " if self.stage_index else ""
        self.message, self.log = prefix + message, ""
        self.update("waiting", detail)

    def update(self, state: str, detail: str, log: Path | None = None) -> None:
        """Args:
            state: waiting, building, complete or failed.
            detail: Human-readable stage/outcome; never an invented percentage.
            log: Current persistent builder log, once the build job is created.
        """
        if not self.active:
            self.start()
        if log is not None:
            self.log = str(log)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(f"{state}\n{self.log}\n{self.message}\n{detail}\nLog: {self.log or '(not started)'}\n", encoding="utf-8", newline="\n")
        os.replace(temporary, self.path)

    def finish(self, error: BaseException | None = None) -> None:
        """Args:
            error: Failure shown with its persistent diagnostic path, if any.
        """
        if self.active:
            self.update("failed" if error else "complete", str(error) if error else "Engine ready; resuming video processing")

    def __enter__(self):
        """Keep the observer alive across cross-process lock waits and compilation."""
        return self

    def __exit__(self, error_type, error, traceback):
        """Args:
            error_type: Exception class from the preparation scope, if any.
            error: Failure retained in the visible status and its persistent log.
            traceback: Python traceback propagated unchanged to the host.
        """
        self.finish(error)
