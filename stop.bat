@echo off
echo ==========================================
echo ОСТАНОВКА МОДЕЛИ ЛЕОНТЬЕВА
echo ==========================================
taskkill /f /im python.exe /fi "WINDOWTITLE eq remote_solver.py" 2>nul
taskkill /f /im streamlit.exe 2>nul
echo Готово!
pause