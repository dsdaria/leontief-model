"""
cpp_bridge.py - связывание Python и C++ библиотекой для решения СЛАУ
"""

import ctypes
import numpy as np
import os
import sys
import time
from typing import Tuple, List, Optional
from pathlib import Path


class CppCGSolver:
    """Класс для работы с C++ DLL/SO решателем"""
    
    _instance = None
    _lib = None
    _is_available = False
    _load_attempted = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_library()
        return cls._instance
    
    def _load_library(self):
        """Загрузка C++ библиотеки (DLL для Windows, SO для Linux)"""
        
        if self._load_attempted:
            return
        
        self._load_attempted = True
        
        print("=" * 60)
        print("🔍 ЗАГРУЗКА C++ БИБЛИОТЕКИ")
        print("=" * 60)
        print(f"🐍 Python версия: {sys.version}")
        print(f"💻 Платформа: {sys.platform}")
        print(f"📁 Текущая директория: {os.getcwd()}")
        
        # Определяем расширение в зависимости от ОС
        if sys.platform == 'win32':
            lib_extension = 'dll'
        else:
            lib_extension = 'so'
        
        # Список возможных путей (приоритет от большего к меньшему)
        possible_paths = [
            # Docker/Render пути (Linux)
            "/app/cg_solver.so",
            "/app/cpp_solver/cg_solver.so",
            "/app/cpp_solver/cg_solver.so",
            # Относительные пути
            os.path.join(os.path.dirname(__file__), "cg_solver.so"),
            os.path.join(os.path.dirname(__file__), "cpp_solver", f"cg_solver.{lib_extension}"),
            os.path.join(os.path.dirname(__file__), f"cg_solver.{lib_extension}"),
            f"./cpp_solver/cg_solver.{lib_extension}",
            f"./cg_solver.{lib_extension}",
            # Windows абсолютные пути
            r"C:\Dasha\Streamlit\leontief-model\cpp_solver\cg_solver.dll",
        ]
        
        # Выводим все пути для отладки
        print("📂 Проверяемые пути:")
        for path in possible_paths:
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"   {exists} {path}")
        
        # Ищем существующий файл
        lib_path = None
        for path in possible_paths:
            if os.path.exists(path):
                lib_path = path
                print(f"\n✅ НАЙДЕНА БИБЛИОТЕКА: {path}")
                break
        
        if lib_path is None:
            print("\n❌ C++ БИБЛИОТЕКА НЕ НАЙДЕНА!")
            print("   Проект будет использовать Python fallback (медленнее)")
            self._lib = None
            self._is_available = False
            print("=" * 60)
            return
        
        # Пробуем загрузить библиотеку
        try:
            print(f"🔧 Загрузка библиотеки: {lib_path}")
            self._lib = ctypes.CDLL(lib_path)
            print("✅ БИБЛИОТЕКА УСПЕШНО ЗАГРУЖЕНА!")
            
            # Проверяем наличие необходимых функций
            required_functions = ['solve_cg', 'solve_batch_cg', 'free_memory']
            missing_functions = []
            
            for func_name in required_functions:
                try:
                    getattr(self._lib, func_name)
                    print(f"   ✅ Функция {func_name} найдена")
                except AttributeError:
                    missing_functions.append(func_name)
                    print(f"   ❌ Функция {func_name} НЕ найдена")
            
            if missing_functions:
                print(f"\n⚠️ ОТСУТСТВУЮТ ФУНКЦИИ: {missing_functions}")
                self._lib = None
                self._is_available = False
                print("=" * 60)
                return
            
            # Определяем типы аргументов для solve_cg
            self._lib.solve_cg.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
            ]
            self._lib.solve_cg.restype = ctypes.POINTER(ctypes.c_double)
            
            # Определяем типы аргументов для solve_batch_cg
            self._lib.solve_batch_cg.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self._lib.solve_batch_cg.restype = ctypes.POINTER(ctypes.c_double)
            
            self._lib.free_memory.argtypes = [ctypes.POINTER(ctypes.c_double)]
            
            self._is_available = True
            print("\n" + "=" * 60)
            print("🎉 C++ РЕШАТЕЛЬ ГОТОВ К РАБОТЕ!")
            print("   • Параллельные вычисления: OpenMP")
            print("   • Метод: сопряжённых градиентов (CG)")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ ОШИБКА ЗАГРУЗКИ БИБЛИОТЕКИ: {e}")
            self._lib = None
            self._is_available = False
            print("=" * 60)
    
    def is_available(self) -> bool:
        return self._is_available and self._lib is not None
    
    def solve_cg(self, A: np.ndarray, b: np.ndarray, 
                 tolerance: float = 1e-8, max_iter: int = 1000,
                 num_threads: int = 4) -> Tuple[np.ndarray, int, float, float, bool]:
        """Решение СЛАУ методом сопряжённых градиентов (C++)"""
        
        if not self.is_available():
            print("⚠️ C++ решатель недоступен, используется Python fallback")
            return self._fallback_solve(A, b, tolerance, max_iter)
        
        print(f"🚀 Используется C++ решатель с {num_threads} потоками")
        
        n = A.shape[0]
        
        from scipy.sparse import csr_matrix
        A_sparse = csr_matrix(A)
        values = A_sparse.data.astype(np.float64)
        col_indices = A_sparse.indices.astype(np.int32)
        row_ptr = A_sparse.indptr.astype(np.int32)
        nnz = len(values)
        
        iterations = ctypes.c_int()
        residual = ctypes.c_double()
        converged = ctypes.c_int()
        
        start = time.perf_counter()
        
        result_ptr = self._lib.solve_cg(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            col_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n, nnz,
            b.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tolerance, max_iter, num_threads,
            ctypes.byref(iterations),
            ctypes.byref(residual),
            ctypes.byref(converged)
        )
        
        elapsed = time.perf_counter() - start
        
        x = np.array([result_ptr[i] for i in range(n)])
        self._lib.free_memory(result_ptr)
        
        return x, iterations.value, residual.value, elapsed, converged.value == 1
    
    def solve_batch_cg(self, A: np.ndarray, B: np.ndarray,
                       tolerance: float = 1e-6, max_iter: int = 2000,
                       num_threads: int = 4) -> Tuple[np.ndarray, float]:
        """Многовариантные расчеты - решение для множества правых частей"""
        
        if not self.is_available():
            raise RuntimeError("C++ solver not available")
        
        n = A.shape[0]
        n_rhs = B.shape[0]
        
        from scipy.sparse import csr_matrix
        A_sparse = csr_matrix(A)
        values = A_sparse.data.astype(np.float64)
        col_indices = A_sparse.indices.astype(np.int32)
        row_ptr = A_sparse.indptr.astype(np.int32)
        nnz = len(values)
        
        b_flat = B.astype(np.float64).flatten()
        
        start = time.perf_counter()
        
        result_ptr = self._lib.solve_batch_cg(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            col_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n, nnz,
            b_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_rhs, tolerance, max_iter, num_threads
        )
        
        elapsed = time.perf_counter() - start
        
        X = np.array([[result_ptr[k * n + i] for i in range(n)] for k in range(n_rhs)])
        self._lib.free_memory(result_ptr)
        
        return X, elapsed
    
    def _fallback_solve(self, A: np.ndarray, b: np.ndarray,
                        tolerance: float = 1e-8, max_iter: int = 1000) -> Tuple[np.ndarray, int, float, float, bool]:
        """Fallback на scipy (если C++ недоступен)"""
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import cg
        
        print("🐍 Используется Python fallback (scipy.sparse.linalg.cg)")
        
        A_sparse = csr_matrix(A)
        start = time.perf_counter()
        x, info = cg(A_sparse, b, rtol=tolerance, maxiter=max_iter)
        elapsed = time.perf_counter() - start
        
        residual = np.linalg.norm(A_sparse @ x - b)
        iterations = info if info >= 0 else max_iter
        converged = info == 0
        
        return x, iterations, residual, elapsed, converged


# Глобальный экземпляр
_solver = None


def get_cg_solver() -> CppCGSolver:
    global _solver
    if _solver is None:
        _solver = CppCGSolver()
    return _solver


def is_cpp_available() -> bool:
    return get_cg_solver().is_available()