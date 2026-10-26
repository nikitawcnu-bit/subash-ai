# Builds dist\SubashAI.exe (Windows, Python 3.10+).
#
#   pip install pyinstaller pillow
#   powershell -ExecutionPolicy Bypass -File app\build_exe.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    Get-Process -Name SubashAI -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 400
    Write-Host "Building SubashAI.exe..."
    python -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name SubashAI `
        --icon "$root\assets\subash.ico" `
        --add-data "$root\checkpoints\gpt_pushkin.npz;checkpoints" `
        --add-data "$root\assets\subash.ico;assets" `
        --paths "$root" `
        --collect-submodules subash `
        --collect-all customtkinter `
        --hidden-import PIL._tkinter_finder `
        --distpath "$root\dist" `
        --workpath "$root\build\pyinstaller" `
        --specpath "$root\build" `
        "$root\app\subash_app.py"

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    $exe = Join-Path $root "dist\SubashAI.exe"
    if (-not (Test-Path $exe)) {
        throw "PyInstaller did not create $exe"
    }
    $size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    Write-Host "Done: $exe ($size MB)"
} finally {
    Pop-Location
}
