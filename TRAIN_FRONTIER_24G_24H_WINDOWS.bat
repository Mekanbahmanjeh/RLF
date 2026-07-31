@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\train_frontier_24g_24h.ps1" %*
exit /b %ERRORLEVEL%
