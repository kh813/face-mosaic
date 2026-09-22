@echo off
setlocal
cd /d "%~dp0"

:: If command-line arguments are provided, run synchronously in CLI mode (in this console)
if "%~1" neq "" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_and_run.ps1" %*
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [Error] Execution stopped with error code %ERRORLEVEL%.
        pause
    )
    exit /b %ERRORLEVEL%
)

:: GUI mode (double-clicked or run without arguments):
:: Launch completely hidden in the background and close cmd.exe immediately.
start "" wscript.exe //nologo "%~dp0scripts\run_gui.vbs"
exit /b 0
