@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\train_preview_6gb.ps1" %*
exit /b %ERRORLEVEL%
