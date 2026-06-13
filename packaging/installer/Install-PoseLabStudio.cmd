@echo off
rem Install PoseLab Studio locally (double-click entry point).
rem Uses only signed cmd.exe / powershell.exe / Python, so Windows
rem Smart App Control does not block it. ASCII + CRLF on purpose.
setlocal
echo Installing PoseLab Studio...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_local.ps1" %*
echo.
pause
