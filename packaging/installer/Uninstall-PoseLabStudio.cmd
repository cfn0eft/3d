@echo off
rem PoseLab Studio uninstaller (double-click entry point).
rem Removes the install directory and shortcuts via uninstall_local.ps1.
rem Pass -Cache to also delete downloaded model weights. Uses only signed
rem cmd / powershell. ASCII + CRLF on purpose.
setlocal
echo.
echo   PoseLab Studio - uninstall
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall_local.ps1" %*
echo.
pause
