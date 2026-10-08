@echo off
rem ===========================================================================
rem  CQU-SRTP Lost and Found - Windows one-click launcher
rem
rem  IMPORTANT: keep this file ASCII-only, WITHOUT a BOM, with CRLF endings.
rem  cmd.exe parses .bat files using the OEM code page (936/GBK on Chinese
rem  Windows), so UTF-8 Chinese text here is decoded as garbage and the mangled
rem  bytes are then executed as bogus commands. All Chinese output lives in
rem  serve.py / bootstrap.py / seed.py instead.
rem
rem  The launcher tries several interpreters in order, because the project
rem  folder may have been moved or renamed after the .venv was created
rem  (a venv stores absolute paths and stops working once moved):
rem      1. .venv\Scripts\python.exe   (if present and healthy)
rem      2. py -3                      (Windows Python launcher)
rem      3. python                     (whatever PATH provides)
rem  Each candidate is probed via "serve.py --probe" (exit code 0 = usable).
rem ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
title CQU-SRTP Lost and Found

set "FOUND="

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" serve.py --probe >nul 2>&1
  if !errorlevel! equ 0 set "FOUND=.venv\Scripts\python.exe"
)

if not defined FOUND (
  py -3 serve.py --probe >nul 2>&1
  if !errorlevel! equ 0 set "FOUND=py -3"
)

if not defined FOUND (
  python serve.py --probe >nul 2>&1
  if !errorlevel! equ 0 set "FOUND=python"
)

if not defined FOUND (
  echo.
  echo [ERROR] No usable Python 3 interpreter was found.
  echo.
  echo   The bundled .venv stores absolute paths, so it stops working after
  echo   the project folder is moved or renamed. Fix it with either:
  echo.
  echo     py -3 serve.py --recreate-venv     needs Python 3 on PATH
  echo     py -3 bootstrap.py                 installs deps into vendor\
  echo.
  echo   Install Python 3.10+ from https://www.python.org/downloads/
  echo   and tick "Add python.exe to PATH" during setup.
  echo.
  pause
  exit /b 1
)

%FOUND% serve.py
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo.
  echo [ERROR] Startup failed with exit code %EXITCODE%
  echo         Try:  %FOUND% bootstrap.py
  echo.
)

pause
exit /b %EXITCODE%
