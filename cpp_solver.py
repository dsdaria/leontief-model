"""
Обёртка для C++ решателя
"""
import ctypes
import numpy as np
from ctypes import c_int, c_double, c_void_p, POINTER
import os

class LeontiefSolver:
    def __init__(self):
        # Загрузка DLL
        dll_path = os.path.join(os.path.dirname(__file__), 'leontief_solver.dll')
        
        try:
            self.lib = ctypes.CDLL(dll_path)
            self._init_functions()
            self.loaded = True
            print("✅ C++ solver loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load C++ solver: {e}")
            self.loaded = False
    
    def _init_functions(self):
        # sparse_matrix_vector_multiply
        self.lib.sparse_matrix_vector_multiply.argtypes = [
            c_int,  # n
            POINTER(c_double),  # data
            POINTER(c_int),  # indices
            POINTER(c_int),  # indptr
            POINTER(c_double),  # x
            POINTER(c_double),  # result
            c_int  # num_threads
        ]
        
        # neumann_solver
        self.lib.neumann_solver.argtypes = [
            c_int,  # n
            POINTER(c_double),  # data
            POINTER(c_int),  # indices
            POINTER(c_int),  # indptr
            POINTER(c_double),  # Y
            POINTER(c_double),  # X
            c_int,  # num_threads
            c_int  # max_iterations
        ]
        
        # gauss_seidel
        self.lib.gauss_seidel.argtypes = [
            c_int,  # n
            POINTER(c_double),  # data
            POINTER(c_int),  # indices
            POINTER(c_int),  # indptr
            POINTER(c_double),  # Y
            POINTER(c_double),  # X
            c_int,  # num_threads
            c_int  # max_iterations
        ]
    
    def solve_neumann(self, A_sparse, Y, num_threads=1, max_iterations=50):
        """Решение методом Неймана"""
        if not self.loaded:
            return None
        
        n = A_sparse.n
        X = np.zeros(n, dtype=np.float64)
        
        # Получаем указатели на данные
        data = A_sparse.data.ctypes.data_as(POINTER(c_double))
        indices = A_sparse.indices.ctypes.data_as(POINTER(c_int))
        indptr = A_sparse.indptr.ctypes.data_as(POINTER(c_int))
        Y_ptr = Y.ctypes.data_as(POINTER(c_double))
        X_ptr = X.ctypes.data_as(POINTER(c_double))
        
        # Вызов C++ функции
        self.lib.neumann_solver(
            n, data, indices, indptr,
            Y_ptr, X_ptr,
            num_threads, max_iterations
        )
        
        return X
    
    def solve_gauss_seidel(self, A_sparse, Y, num_threads=1, max_iterations=200):
        """Решение методом Гаусса-Зейделя"""
        if not self.loaded:
            return None
        
        n = A_sparse.n
        X = np.zeros(n, dtype=np.float64)
        
        data = A_sparse.data.ctypes.data_as(POINTER(c_double))
        indices = A_sparse.indices.ctypes.data_as(POINTER(c_int))
        indptr = A_sparse.indptr.ctypes.data_as(POINTER(c_int))
        Y_ptr = Y.ctypes.data_as(POINTER(c_double))
        X_ptr = X.ctypes.data_as(POINTER(c_double))
        
        self.lib.gauss_seidel(
            n, data, indices, indptr,
            Y_ptr, X_ptr,
            num_threads, max_iterations
        )
        
        return X


# Глобальный экземпляр
_solver = None

def get_solver():
    global _solver
    if _solver is None:
        _solver = LeontiefSolver()
    return _solver