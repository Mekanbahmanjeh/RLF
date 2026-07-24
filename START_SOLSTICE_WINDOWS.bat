@echo off
setlocal
set "EXE=%~dp0build\windows-msvc-release\Release\solstice.exe"
set "MODEL=%~dp0models\solstice_preview_sample.rlfsp"
if not exist "%EXE%" (
  echo Solstice has not been built yet.
  echo Run BUILD_WINDOWS.bat first.
  exit /b 1
)
"%EXE%" chat --checkpoint "%MODEL%" %*
exit /b %ERRORLEVEL%
