"""
Параллельное решение СЛАУ (I - A)X = Y
"""

import numpy as np
from scipy.sparse.linalg import bicgstab, gmres
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import time
from typing import List, Tuple, Dict


class ParallelSLESolver:
    """
    Параллельное решение систем линейных алгебраических уравнений
    с несколькими правыми частями
    """
    
    def __init__(self, I_minus_A: np.ndarray, tol: float = 1e-8, maxiter: int = 1000):
        """
        Args:
            I_minus_A: матрица (I - A)
            tol: точность итерационных методов
            maxiter: максимум итераций
        """
        self.n = I_minus_A.shape[0]
        self.I_minus_A = I_minus_A
        self.tol = tol
        self.maxiter = maxiter
        
        # Для больших матриц используем разреженное представление
        if self.n > 50:
            self.A_sparse = csr_matrix(I_minus_A)
        else:
            self.A_sparse = I_minus_A
    
    def solve_single(self, Y: np.ndarray, method: str = 'direct') -> np.ndarray:
        """
        Решение для одной правой части
        
        Args:
            Y: вектор правой части
            method: 'direct' - прямой метод solve, 'bicgstab' или 'gmres'
        """
        if method == 'direct':
            return np.linalg.solve(self.I_minus_A, Y)
        elif method == 'bicgstab':
            x, info = bicgstab(self.A_sparse, Y, tol=self.tol, maxiter=self.maxiter)
            return x if info == 0 else np.linalg.solve(self.I_minus_A, Y)
        elif method == 'gmres':
            x, info = gmres(self.A_sparse, Y, tol=self.tol, maxiter=self.maxiter)
            return x if info == 0 else np.linalg.solve(self.I_minus_A, Y)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def solve_batch_parallel(self, Y_list: List[np.ndarray], 
                             method: str = 'bicgstab',
                             n_jobs: int = -1) -> List[np.ndarray]:
        """
        ПАРАЛЛЕЛЬНОЕ решение для МНОГИХ правых частей
        Это ключевая функция для Этапа 3!
        
        Args:
            Y_list: список векторов правых частей
            method: метод решения
            n_jobs: количество потоков (-1 = все доступные)
        
        Returns:
            список решений X
        """
        def solve_one(Y):
            return self.solve_single(Y, method)
        
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(solve_one)(Y) for Y in Y_list
        )
        
        return results
    
    def batch_solve_direct(self, Y_matrix: np.ndarray) -> np.ndarray:
        """
        Решение для МАТРИЦЫ правых частей (эффективнее, чем по одной)
        Использует BLAS/LAPACK с многопоточностью
        
        Args:
            Y_matrix: матрица (n, k) — k правых частей
        
        Returns:
            матрица решений (n, k)
        """
        # np.linalg.solve автоматически использует многопоточный LAPACK
        return np.linalg.solve(self.I_minus_A, Y_matrix)
    
    def benchmark_batch(self, n_rhs: int = 100) -> Dict:
        """
        Бенчмарк: сравнение последовательного и параллельного решения
        
        Args:
            n_rhs: количество правых частей
        """
        # Генерируем случайные правые части
        np.random.seed(42)
        Y_list = [np.random.randn(self.n) for _ in range(n_rhs)]
        Y_matrix = np.column_stack(Y_list)
        
        # Метод 1: Пакетное решение через solve (многопоточный LAPACK)
        start = time.perf_counter()
        X_batch = self.batch_solve_direct(Y_matrix)
        batch_time = time.perf_counter() - start
        
        # Метод 2: Последовательное решение по одному
        start = time.perf_counter()
        X_seq = [self.solve_single(Y, 'direct') for Y in Y_list]
        seq_time = time.perf_counter() - start
        
        # Метод 3: Параллельное решение через joblib
        start = time.perf_counter()
        X_par = self.solve_batch_parallel(Y_list, method='direct', n_jobs=-1)
        par_time = time.perf_counter() - start
        
        return {
            'n_rhs': n_rhs,
            'matrix_size': self.n,
            'batch_direct_time': batch_time,
            'sequential_time': seq_time,
            'parallel_joblib_time': par_time,
            'speedup_parallel_vs_seq': seq_time / par_time if par_time > 0 else 0,
            'speedup_batch_vs_seq': seq_time / batch_time if batch_time > 0 else 0
        }


# ===================== ФУНКЦИЯ ДЛЯ ИСПОЛЬЗОВАНИЯ В МОДЕЛИ =====================

def solve_leontief_system(I_minus_A: np.ndarray, 
                          Y: np.ndarray,
                          method: str = 'parallel_batch') -> np.ndarray:
    """
    Решение системы (I - A)X = Y
    
    Args:
        I_minus_A: матрица (I - A)
        Y: вектор или матрица правых частей
        method: 
            - 'direct': прямой метод solve
            - 'iterative': итерационный метод
            - 'parallel_batch': пакетный параллельный (рекомендуется)
    """
    n = I_minus_A.shape[0]
    
    if method == 'parallel_batch' and Y.ndim == 2:
        # Пакетное решение через многопоточный LAPACK
        return np.linalg.solve(I_minus_A, Y)
    elif method == 'parallel_batch':
        return np.linalg.solve(I_minus_A, Y.reshape(-1, 1)).flatten()
    elif method == 'direct':
        return np.linalg.solve(I_minus_A, Y)
    elif method == 'iterative':
        solver = ParallelSLESolver(I_minus_A)
        return solver.solve_single(Y, 'bicgstab')
    else:
        raise ValueError(f"Неизвестный метод: {method}")