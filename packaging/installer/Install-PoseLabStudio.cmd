@echo off
rem PoseLab Studio installer / updater (double-click entry point).
rem Pulls the latest version (git), then installs via install_local.ps1.
rem Uses only signed cmd / git / powershell / Python, so Windows Smart App
rem Control does not block it. ASCII + CRLF on purpose. Keep this file stable
rem so a self-update during git pull stays safe.
setlocal
echo.
echo   PoseLab Studio - update and install
echo   Fetching the latest version...
pushd "%~dp0..\.."
git pull --ff-only || echo   (Could not update automatically; using the current version.)
popd
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_local.ps1" %*
echo.
pause
