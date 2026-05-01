#!/bin/bash
# stop.sh - Остановка проекта

echo "=========================================="
echo "🛑 ОСТАНОВКА МОДЕЛИ ЛЕОНТЬЕВА"
echo "=========================================="

# Остановка на порту 5000 (Flask)
for pid in $(lsof -ti:5000 2>/dev/null); do
    kill -9 $pid 2>/dev/null
    echo "Остановлен PID: $pid (порт 5000)"
done

# Остановка на порту 8501 (Streamlit)
for pid in $(lsof -ti:8501 2>/dev/null); do
    kill -9 $pid 2>/dev/null
    echo "Остановлен PID: $pid (порт 8501)"
done

# Остановка python процессов
pkill -f "remote_solver.py" 2>/dev/null
pkill -f "streamlit run" 2>/dev/null

echo "✅ Все процессы остановлены"