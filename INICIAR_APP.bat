@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
where py >nul 2>nul
if errorlevel 1 goto python_fallback
py -3.12 -m venv .venv
if errorlevel 1 goto error
goto install
:python_fallback
python -m venv .venv
if errorlevel 1 goto error
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
:run
".venv\Scripts\python.exe" -m streamlit run app.py
goto end
:error
echo No se pudo preparar el entorno. Abre README.md y sigue los pasos con Python 3.12 o Anaconda.
pause
:end
endlocal
