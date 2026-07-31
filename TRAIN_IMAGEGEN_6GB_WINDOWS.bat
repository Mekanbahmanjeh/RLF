@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\train_imagegen_6gb.ps1" %*
exit /b %ERRORLEVEL%
