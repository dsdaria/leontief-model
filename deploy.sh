#!/bin/bash
# deploy.sh - Деплой на GitHub

echo "=========================================="
echo "🚀 Деплой Leontief Model на GitHub"
echo "=========================================="

cd "$(dirname "$0")"

# Проверка git
if ! command -v git &> /dev/null; then
    echo "❌ Git не установлен!"
    exit 1
fi

# Удаление ненужных файлов
rm -rf outputs/ data/ __pycache__/ 2>/dev/null

# Добавление файлов
git add .

# Коммит
echo "Введите сообщение коммита:"
read commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Update: Leontief model"
fi
git commit -m "$commit_msg"

# Пуш
git push origin main

echo "✅ Деплой завершён!"