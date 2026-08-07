$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$venvDir = Join-Path $repoRoot ".venv-cpu"
$pipCache = Join-Path $repoRoot "tmp\pip-cache"
$pipTemp = Join-Path $repoRoot "tmp\pip-temp"
$nltkData = Join-Path $repoRoot ".nltk_data"

New-Item -ItemType Directory -Force -Path $pipCache, $pipTemp, $nltkData | Out-Null
if (-not (Test-Path $venvDir)) {
    python -m venv $venvDir
}

$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$env:PIP_CACHE_DIR = $pipCache
$env:TEMP = $pipTemp
$env:TMP = $pipTemp
$env:NLTK_DATA = $nltkData

& $pythonExe -m pip install --upgrade pip wheel setuptools
& $pythonExe -m pip install --no-build-isolation -r (Join-Path $repoRoot "requirements-cpu-test.txt")
& $pythonExe -c "import nltk; nltk.download('punkt', download_dir=r'$nltkData'); nltk.download('punkt_tab', download_dir=r'$nltkData')"

Write-Host "CPU verification environment ready: $venvDir"
Write-Host "Run: `$env:NLTK_DATA='$nltkData'; & '$pythonExe' -m unittest discover -s tests -v"
