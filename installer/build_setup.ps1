# Builds dist\SubashAI-Setup-0.2.0.exe with a portable NSIS compiler.
# Run app\build_exe.ps1 first.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$nsis = Join-Path $root "tools\nsis-3.10\makensis.exe"
$zip = Join-Path $env:TEMP "nsis-3.10.zip"

if (-not (Test-Path $nsis)) {
    Write-Host "Downloading portable NSIS 3.10..."
    New-Item -ItemType Directory -Force (Join-Path $root "tools") | Out-Null
    $url = "https://downloads.sourceforge.net/project/nsis/NSIS%203/3.10/nsis-3.10.zip"
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        curl.exe -L --fail --retry 2 -o $zip $url
        if ($LASTEXITCODE -ne 0) { throw "Could not download NSIS" }
    } else {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    }
    Expand-Archive -Path $zip -DestinationPath (Join-Path $root "tools") -Force
}

if (-not (Test-Path $nsis)) {
    throw "NSIS compiler not found at $nsis"
}

$appExe = Join-Path $root "dist\SubashAI.exe"
if (-not (Test-Path $appExe)) {
    throw "Build the app first: powershell -ExecutionPolicy Bypass -File app\build_exe.ps1"
}

& $nsis "-V2" "-INPUTCHARSET" "UTF8" (Join-Path $root "installer\subash.nsi")
if ($LASTEXITCODE -ne 0) {
    throw "makensis failed with exit code $LASTEXITCODE"
}

$setup = Join-Path $root "dist\SubashAI-Setup-0.3.0.exe"
if (-not (Test-Path $setup)) {
    throw "Installer was not created: $setup"
}
$size = [math]::Round((Get-Item $setup).Length / 1MB, 1)
Write-Host "Done: $setup ($size MB)"
