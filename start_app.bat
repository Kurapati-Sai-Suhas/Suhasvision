@echo off
echo Starting Django Backend (Ollama Vision)...
start cmd /k "cd backend\backend && ..\venv\Scripts\activate && python manage.py runserver"

echo Starting Vite Frontend...
start cmd /k "cd Frontend && npm run dev"

echo Both servers are starting up. Ensure Ollama is running (`ollama serve`).
