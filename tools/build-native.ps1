param(
    [Parameter(Mandatory = $true)][string]$TensorRTRoot,
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$CudaArchitectures = '75-real;80-real;86-real;89-real;90-real;100-real;120-real;75-virtual',
    [switch]$ProgressOnly
)
# TensorRTRoot: extracted official SDK; CudaRoot: installed CUDA Toolkit;
# CudaArchitectures: semicolon-separated NVIDIA SM targets for this local build.
# ProgressOnly: rebuild/stage the console observer, preserving inference binaries.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetDirectoryName($PSScriptRoot)
$taskSdk = (Resolve-Path -LiteralPath $TensorRTRoot).Path
if (-not $CudaRoot) { throw 'Provide -CudaRoot or set CUDA_PATH in this build process.' }
$taskCuda = (Resolve-Path -LiteralPath $CudaRoot).Path
$taskVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$taskVs = (& $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $taskVs) { throw 'MSVC x64 build tools are required for development.' }
$taskDev = Join-Path $taskVs 'Common7/Tools/VsDevCmd.bat'
# Import developer variables into this build process only; no filesystem shell work.
$taskLines = & $env:ComSpec /d /s /c ('"' + $taskDev + '" -no_logo -arch=amd64 -host_arch=amd64 >nul && set')
if ($LASTEXITCODE -ne 0) { throw 'VsDevCmd failed.' }
foreach ($taskLine in $taskLines) {
    $taskEqual = $taskLine.IndexOf('=')
    if ($taskEqual -gt 0) {
        [Environment]::SetEnvironmentVariable($taskLine.Substring(0, $taskEqual), $taskLine.Substring($taskEqual + 1), 'Process')
    }
}
$taskBuild = Join-Path $taskRoot 'build/native'
$taskCmake = (Get-Command cmake.exe -ErrorAction Stop).Source
$taskNvcc = Join-Path $taskCuda 'bin/nvcc.exe'
$taskCl = (Get-Command cl.exe -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $taskBuild -Force | Out-Null
& $taskCmake -S (Join-Path $taskRoot 'native') -B $taskBuild -G 'NMake Makefiles' `
    '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_CXX_COMPILER=$taskCl" "-DCMAKE_CUDA_HOST_COMPILER=$taskCl" `
    "-DCMAKE_CUDA_COMPILER=$taskNvcc" "-DCUDAToolkit_ROOT=$taskCuda" "-DTENSORRT_ROOT=$taskSdk" `
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" 2>&1 | Tee-Object -FilePath (Join-Path $taskBuild 'configure.log')
if ($LASTEXITCODE -ne 0) { throw 'Native configure failed; see build/native/configure.log.' }
$taskTargets = if ($ProgressOnly) { @('--target', 'rsb_progress') } else { @() }
& $taskCmake --build $taskBuild @taskTargets 2>&1 | Tee-Object -FilePath (Join-Path $taskBuild 'build.log')
if ($LASTEXITCODE -ne 0) { throw 'Native build failed; see build/native/build.log.' }
$taskPackage = Join-Path $taskRoot 'packages/backend-tensorrt/src/rife_superboost_tensorrt/_native'
New-Item -ItemType Directory -Path $taskPackage -Force | Out-Null
$taskNames = if ($ProgressOnly) { @('rsb_progress.exe') } else { @('rsb_runtime.dll', 'rsb_builder.exe', 'rsb_progress.exe') }
foreach ($taskName in $taskNames) {
    $taskInput = Join-Path $taskBuild "bin/$taskName"
    $taskOutput = Join-Path $taskPackage $taskName
    if (-not (Test-Path -LiteralPath $taskOutput) -or
        (Get-FileHash -LiteralPath $taskInput -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256).Hash) {
        Copy-Item -LiteralPath $taskInput -Destination $taskOutput -Force
    }
}
$taskInfo = @{
    schema_version = 1
    tensorrt = '10.16.1.11'
    cuda = '13.2'
    cuda_architectures = $CudaArchitectures.Split(';')
    msvc_runtime = 'static'
    cuda_runtime = 'static'
    runtime_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskPackage 'rsb_runtime.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
    builder_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskPackage 'rsb_builder.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
    progress_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskPackage 'rsb_progress.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
}
$taskInfo | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskPackage 'build-info.json') -Encoding utf8
Write-Host "Built and staged private native binaries: $taskPackage"
