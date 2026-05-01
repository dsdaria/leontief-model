@echo off
echo ==========================================
echo ЗАПУСК МОДЕЛИ ЛЕОНТЬЕВА
echo ==========================================
cd /d C:\BMSTU\OS\leontief-model
start cmd /k "python remote_solver.py"
timeout /t 2
start cmd /k "streamlit run app.py"
echo Готово! Откройте http://localhost:8501
pause