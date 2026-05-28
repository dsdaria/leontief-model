"""
Бенчмарк: сравнение производительности разных методов решения СЛАУ
"""

import numpy as np
import time
import pandas as pd
from typing import Dict
from parallel_solver import ParallelSLESolver


def generate_test_matrix(n: int, seed: int = 42) -> np.ndarray:
    """Генерация продуктивной матрицы (I - A)"""
    np.random.seed(seed)
    A = np.random.rand(n, n) * 0.5 / n  # Разреженная
    # Нормализуем для продуктивности (спектральный радиус < 1)
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = max(abs(eigenvalues))
    if spectral_radius >= 1:
        A = A * 0.9 / spectral_radius
    return np.eye(n) - A


def run_solver_benchmark(sizes: list = [64, 128, 256, 512],
                         n_rhs: int = 50) -> pd.DataFrame:
    """
    Запуск бенчмарка для разных размеров матриц
    
    Args:
        sizes: список размеров матриц
        n_rhs: количество правых частей
    """
    results = []
    
    for n in sizes:
        print(f"\n📊 Тестирование матрицы {n}×{n}...")
        
        # Генерируем матрицу и правые части
        I_minus_A = generate_test_matrix(n)
        np.random.seed(42)
        Y_list = [np.random.randn(n) for _ in range(n_rhs)]
        Y_matrix = np.column_stack(Y_list)
        
        solver = ParallelSLESolver(I_minus_A)
        
        # === Тест 1: Пакетное решение (многопоточный LAPACK) ===
        start = time.perf_counter()
        X_batch = solver.batch_solve_direct(Y_matrix)
        batch_time = time.perf_counter() - start
        
        # === Тест 2: Последовательное решение ===
        start = time.perf_counter()
        X_seq = []
        for Y in Y_list:
            X_seq.append(solver.solve_single(Y, 'direct'))
        seq_time = time.perf_counter() - start
        
        # === Тест 3: Параллельное joblib ===
        start = time.perf_counter()
        X_par = solver.solve_batch_parallel(Y_list, method='direct', n_jobs=-1)
        par_time = time.perf_counter() - start
        
        results.append({
            'Размер матрицы': n,
            'Правых частей': n_rhs,
            'Пакетный (batch)': round(batch_time, 4),
            'Последовательный': round(seq_time, 4),
            'Параллельный (joblib)': round(par_time, 4),
            'Ускорение batch': round(seq_time / batch_time, 2),
            'Ускорение joblib': round(seq_time / par_time, 2)
        })
    
    return pd.DataFrame(results)


def compare_methods_for_leontief(n: int = 64) -> Dict:
    """
    Сравнение методов для модели Леонтьева
    """
    I_minus_A = generate_test_matrix(n)
    
    # Генерируем 100 сценариев
    np.random.seed(42)
    scenarios = [np.random.randn(n) * 100 for _ in range(100)]
    
    solver = ParallelSLESolver(I_minus_A)
    
    # Метод A: инверсия + умножение
    start = time.perf_counter()
    L = np.linalg.inv(I_minus_A)
    results_inv = [L @ delta for delta in scenarios]
    inv_time = time.perf_counter() - start
    
    # Метод B: пакетное решение
    Y_matrix = np.column_stack(scenarios)
    start = time.perf_counter()
    X_batch = np.linalg.solve(I_minus_A, Y_matrix)
    batch_time = time.perf_counter() - start
    
    # Метод C: параллельное решение (joblib)
    start = time.perf_counter()
    results_par = solver.solve_batch_parallel(scenarios, method='direct', n_jobs=-1)
    par_time = time.perf_counter() - start
    
    return {
        'n': n,
        'n_scenarios': 100,
        'inv_method_time': inv_time,
        'batch_solve_time': batch_time,
        'parallel_joblib_time': par_time,
        'speedup_batch_vs_inv': inv_time / batch_time,
        'speedup_joblib_vs_seq': inv_time / par_time,
        'recommendation': 'batch_solve' if batch_time < inv_time else 'inverse'
    }


if __name__ == "__main__":
    print("=" * 60)
    print("БЕНЧМАРК РЕШЕНИЯ СЛАУ ДЛЯ МОДЕЛИ ЛЕОНТЬЕВА")
    print("=" * 60)
    
    # Малые матрицы (как в Eurostat)
    print("\n📊 Малые матрицы (64×64):")
    small_results = compare_methods_for_leontief(64)
    for k, v in small_results.items():
        print(f"  {k}: {v}")
    
    # Большие матрицы (как в EXIOBASE)
    print("\n📊 Большие матрицы (200×200):")
    large_results = compare_methods_for_leontief(200)
    for k, v in large_results.items():
        print(f"  {k}: {v}")