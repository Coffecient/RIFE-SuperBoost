param([string]$Python)
# Python: development interpreter with build, setuptools>=78 and wheel installed.
# Standard PEP 517 builds consume pre-staged native/model assets. No SDK, model
# export, compiler, or network operation is invoked by a package build hook.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetDirectoryName($PSScriptRoot)
if (-not $Python) { $Python = Join-Path $taskRoot '.venv/Scripts/python.exe' }
$taskPython = (Resolve-Path -LiteralPath $Python).Path
$taskDist = Join-Path $taskRoot 'dist'
$taskLogs = Join-Path $taskRoot 'build'
New-Item -ItemType Directory -Path $taskDist, $taskLogs -Force | Out-Null
foreach ($taskPackage in @('core', 'backend-tensorrt', 'models/4.25', 'models/4.25-lite', 'models/4.26', 'models/4.26-large', 'models/4.25-scale8', 'models/4.26-scale8')) {
    $taskLog = Join-Path $taskLogs ('wheel-' + $taskPackage.Replace('/', '-') + '.log')
    & $taskPython -m build --sdist --wheel --no-isolation --outdir $taskDist (Join-Path $taskRoot "packages/$taskPackage") *> $taskLog
    if ($LASTEXITCODE -ne 0) { throw "PEP 517 build failed for $taskPackage; see $taskLog." }
}
Get-ChildItem -LiteralPath $taskDist -Filter '*.whl' | Select-Object Name, Length
