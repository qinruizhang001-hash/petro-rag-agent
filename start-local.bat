@echo off
rem ---------------------------------------------------------------------------
rem Start the CCUS console locally: Ollama -> backend -> web preview.
rem Double-click this file. Three console windows stay open so you can read logs.
rem
rem Why a .bat exists next to start-local.sh: the .sh needs Git Bash. This one
rem only needs what Windows already has. Output is ASCII on purpose - a .bat
rem written in UTF-8 Chinese is displayed as garbage under the default GBK
rem codepage, and "garbage" is indistinguishable from "the script is broken".
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "PY=D:\python-learn\.venv\Scripts\python.exe"
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
set "THR=config\ops_thresholds.example.json"
set "URL=http://127.0.0.1:4173/?source=live"

echo.
echo === CCUS console / local start ===
echo.

if not exist "%PY%" (
  echo [X] Python not found: %PY%
  echo     Set PY at the top of this file to your venv python.exe
  pause
  exit /b 1
)
if not exist "%THR%" (
  echo [X] Threshold file not found: %THR%
  pause
  exit /b 1
)

rem --- why --noproxy "*" and -f on every probe -------------------------------
rem If http_proxy is set, curl routes even 127.0.0.1 through the proxy. A dead
rem local port then comes back as the proxy's 502, and curl exits 0 for a 502 -
rem so "nothing is listening" reads as "already running" and the script skips
rem the start, prints the URL, and the user opens a blank page.
rem ---------------------------------------------------------------------------
rem --- 1/3 Ollama -------------------------------------------------------------
echo [1/3] Ollama
curl -sf --noproxy "*" -o nul -m 2 http://127.0.0.1:11434/api/tags
if errorlevel 1 (
  if exist "%OLLAMA%" (
    start "ccus-ollama" cmd /k ""%OLLAMA%" serve"
    echo   starting ^(retrieval and embeddings need it^)
  ) else (
    echo   [!] ollama.exe not found - the console will open, but search/chat fail
  )
) else (
  echo   already running
)

rem --- 2/3 backend ------------------------------------------------------------
rem 15 s, not 3: /health probes four dependencies one after another and the
rem Ollama check alone takes ~2 s. A 3 s timeout turns a healthy backend into
rem "not running", the script starts a second uvicorn, the port is taken, the
rem new process dies - and the wait still succeeds because the OLD process
rem answers. Everything looks fine and there is a dead process behind it.
echo [2/3] backend
curl -sf --noproxy "*" -o nul -m 15 http://127.0.0.1:8000/health
if errorlevel 1 (
  rem First start loads a 3.95 GB reranker. ~15 s is normal; do not assume it hung.
  rem Set the threshold path in THIS shell and let the child inherit it, instead
  rem of nesting "set X && command" inside one cmd /k string - there a single
  rem misplaced quote turns the whole line into "The system cannot find the file
  rem specified", with the window closing before anyone can read it.
  set "OPS_THRESHOLDS_PATH=%THR%"
  rem No inner quotes around %PY%: the default path has no spaces, so the whole
  rem command stays plain and cmd has no quote-stripping rule to get wrong. If
  rem you point PY at a path WITH spaces, write it as cmd /k ""%PY%" -m ...".
  start "ccus-backend" cmd /k "%PY%" -m uvicorn api_server:api --host 127.0.0.1 --port 8000
  echo   starting ^(first run loads a 3.95 GB reranker, ~15 s^)
) else (
  echo   already running
)

rem --- 3/3 web ----------------------------------------------------------------
echo [3/3] web
curl -sf --noproxy "*" -o nul -m 3 http://127.0.0.1:4173/
if errorlevel 1 (
  if not exist "web\dist" (
    echo   web\dist missing - building first ^(about 10 s^)
    pushd web
    call npm run build
    popd
  )
  start "ccus-web" cmd /k "cd web && npx vite preview --port 4173 --host 127.0.0.1"
  echo   starting
) else (
  echo   already running
)

echo.
echo Waiting for the backend to finish loading ...
rem Poll /health instead of sleeping a fixed time: on a cold file cache the
rem reranker takes noticeably longer, and opening the page too early shows
rem "backend unreachable" - which looks like a failure but is just impatience.
set /a tries=0
:waitloop
curl -sf --noproxy "*" -m 15 http://127.0.0.1:8000/health | findstr /c:"\"status\"" >nul
if not errorlevel 1 goto ready
set /a tries+=1
if %tries% geq 60 goto timeout
timeout /t 1 /nobreak >nul
goto waitloop

:ready
echo.
echo === ready ===
echo.
echo   %URL%
echo.
echo   ?source=live   real backend (readings are a replay of the captured sample)
echo   no parameter   snapshot only - opens even with the backend down
echo.
echo   ?mode=engineer  engineer mode (cards / table)
echo   ?tab=chain      full chain view
echo   ?rail=dialog    chat panel
echo   ?kb=search      knowledge base search
echo.

rem Voice: ask the download script rather than re-testing the model files here.
rem The model name and "which files count as complete" must have one definition.
"%PY%" scripts\fetch_asr_model.py --check >nul 2>&1
if errorlevel 1 (
  echo   voice  local model NOT downloaded - voice still works, but it falls
  echo          back to the browser's cloud recogniser and the recording
  echo          leaves this machine. Get the local model with:
  echo            "%PY%" scripts\fetch_asr_model.py
) else (
  echo   voice  local ASR ready - audio never leaves this machine
)
echo.
start "" "%URL%"
exit /b 0

:timeout
echo.
echo [!] backend did not report /health within 60 s.
echo     Check the "ccus-backend" window for the real error.
echo     Common causes: Ollama not running (status=degraded is fine to use,
echo     search will fail), or config\ops_thresholds.example.json missing.
pause
exit /b 1
