$ErrorActionPreference = 'Stop'
$taskMainPython = Join-Path $PSScriptRoot '..\asmr-next\.venv\Scripts\python.exe'
$taskTorchPython = Join-Path $PSScriptRoot '..\asmr-next\.asmr-dubber\runtimes\index-tts\.venv\Scripts\python.exe'
$taskUv = Join-Path $PSScriptRoot '..\asmr-next\.asmr-dubber\bootstrap\windows\uv\uv.exe'
$env:PYTHONUTF8 = '1'
& $taskMainPython (Join-Path $PSScriptRoot 'setup_spatial.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskUv pip install --python $taskTorchPython --target (Join-Path $PSScriptRoot 'work\spatial-deps') --no-deps h5py==3.14.0
exit $LASTEXITCODE
