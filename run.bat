@echo off
echo ================================================
echo   NEURAL STUDY OS - Launcher
echo ================================================
echo.
echo Step 1: Starting Ollama AI server...
echo If this fails, install from https://ollama.com
echo.
start ollama serve
ping localhost -n 5 > nul
echo Step 2: Installing Python dependencies...
pip install -r requirements.txt
echo Step 3: Starting Neural Study OS...
echo Open http://localhost:5000 in your browser
echo.
python main.py
pause
