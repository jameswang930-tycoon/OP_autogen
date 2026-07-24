@echo off
REM =============================================================================
REM Triton Agent Optimizer — Windows 本地环境设置 + 验证
REM =============================================================================
REM  用法:
REM   prepare\setup_env.bat             设置环境 + 验证
REM   prepare\setup_env.bat --verify    只验证
REM =============================================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set AGENT_DIR=%SCRIPT_DIR%..
set PROJECT_DIR=%AGENT_DIR%\..\..

echo ==============================================
echo  Triton Agent Optimizer — Windows Setup
echo  Agent: %AGENT_DIR%
echo ==============================================

REM ── 1. 检测 Python ──────────────────────────────────────────────────────

echo.
echo --- Python ---
set PYTHON=
for %%p in (python python3) do (
    where %%p >nul 2>nul
    if !errorlevel!==0 (
        set PYTHON=%%p
        goto :found_python
    )
)
echo [ERROR] Python not found
goto :verify

:found_python
%PYTHON% --version
echo Path: %PYTHON%

REM 检查必要包
echo Checking packages...
%PYTHON% -c "import numpy" 2>nul && echo   numpy: OK || echo   numpy: INSTALLING... && pip install numpy -q
%PYTHON% -c "import matplotlib" 2>nul && echo   matplotlib: OK || echo   matplotlib: not installed (optional)
%PYTHON% -c "import networkx" 2>nul && echo   networkx: OK || echo   networkx: not installed (optional)

REM ── 2. 检查 Emulator ───────────────────────────────────────────────────

echo.
echo --- CPU Emulator ---
%PYTHON% -c "import sys; sys.path.insert(0,'%PROJECT_DIR%/emulators'); from common import tl; print('  tl class: OK')" 2>nul || echo   [WARN] emulator not available

REM ── 3. 检查 Simulator ──────────────────────────────────────────────────

echo.
echo --- Cost Simulator ---
set SIM=%PROJECT_DIR%\costModel\cost_emulator\simulator.py
if exist "%SIM%" (
    echo   path: %SIM%
    %PYTHON% "%SIM%" --llm "alloc(gm_1, 1KB) alloc(ub_1, 1KB) gm_to_ub(ub_1, gm_1)" >nul 2>&1 && echo   --llm mode: OK || echo   --llm mode: FAIL
) else (
    echo   [WARN] simulator not found: %SIM%
)

REM ── 4. 检查 Ascend (910B3 only) ──────────────────────────────────────

echo.
echo --- Ascend/CANN ---
where msprof >nul 2>nul && echo   msprof: found || echo   msprof: not found (normal on Windows)
where npu-smi >nul 2>nul && echo   npu-smi: found || echo   npu-smi: not found (normal on Windows)
if defined ASCEND_HOME (echo   ASCEND_HOME: %ASCEND_HOME%) else (echo   ASCEND_HOME: not set (normal on Windows))

REM ── 5. 设置环境变量 ──────────────────────────────────────────────────

echo.
echo --- Environment Variables ---
set PYTHONPATH=%AGENT_DIR%;%PYTHONPATH%
echo   PYTHONPATH += %AGENT_DIR%
echo   Project root: %PROJECT_DIR%

REM ── 6. 验证 ──────────────────────────────────────────────────────────

:verify
echo.
echo ==============================================
echo  Verification
echo ==============================================

cd /d "%AGENT_DIR%"

if "%1"=="--json" (
    %PYTHON% prepare\env_check.py --json
) else if "%1"=="--verify" (
    %PYTHON% prepare\env_check.py
) else (
    %PYTHON% prepare\env_check.py
)

echo.
echo ==============================================
echo  Setup complete.
echo  Run: cd triton_agent_optimizer
echo       python agents/orchestrator.py
echo ==============================================

endlocal
