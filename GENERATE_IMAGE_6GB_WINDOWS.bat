@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\generate_image_6gb.ps1" %*
exit /b %ERRORLEVEL%
