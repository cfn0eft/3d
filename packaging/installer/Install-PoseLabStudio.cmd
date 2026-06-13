@echo off
rem PoseLab Studio をローカルにインストールする (ダブルクリック用)。
rem 署名済みの cmd.exe / powershell.exe / Python しか使わないため、
rem Smart App Control にブロックされない。
setlocal
echo PoseLab Studio をインストールします...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_local.ps1" %*
echo.
pause
