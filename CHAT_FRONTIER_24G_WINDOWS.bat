@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\chat_frontier_24g.ps1" %*
exit /b %ERRORLEVEL%
