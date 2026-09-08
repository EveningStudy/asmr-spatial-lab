$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '..\asmr-next\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw '找不到相邻 asmr-next 的 Python 环境。请用已安装 requirements.txt 的 Python 运行 dub.py。'
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $taskPython (Join-Path $PSScriptRoot 'dub.py') @args
exit $LASTEXITCODE
