"""
test_cpp_simple.py - Простая проверка C++ DLL
"""

import ctypes
import os
import numpy as np

# Путь к DLL
dll_path = r"C:\Dasha\Streamlit\leontief-model\cpp_solver\cg_solver.dll"

print("=" * 60)
print("Проверка C++ DLL")
print("=" * 60)

# Проверяем, существует ли файл
if os.path.exists(dll_path):
    print(f"✅ DLL найдена: {dll_path}")
    print(f"   Размер: {os.path.getsize(dll_path)} bytes")
else:
    print(f"❌ DLL не найдена!")
    exit(1)

# Пробуем загрузить DLL
try:
    lib = ctypes.CDLL(dll_path)
    print("✅ DLL загружена успешно!")
    
    # Проверяем, что функции экспортированы
    functions = ['solve_cg', 'solve_batch_cg', 'free_memory', 'free_int_memory']
    
    for func_name in functions:
        try:
            getattr(lib, func_name)
            print(f"   ✅ Функция {func_name} найдена")
        except AttributeError:
            print(f"   ❌ Функция {func_name} не найдена")
    
    print("\n✅ C++ библиотека готова к использованию!")
    
except Exception as e:
    print(f"❌ Ошибка загрузки DLL: {e}")