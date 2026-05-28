"""
parallel_solver.py - Параллельное решение СЛАУ (I - A)X = Y
1. Прямое решение без полной инверсии матрицы
2. Параллельное решение для множества правых частей
3. Сравнение методов (direct, iterative, batch)
"""

import numpy as np
from scipy.sparse.linalg import bicgstab, gmres
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import time
from typing import List, Tuple, Dict, Union
import warnings
warnings.filterwarnings('ignore')


class ParallelSLESolver:
    """
    Параллельное решение систем линейных алгебраических уравнений
    для модели Леонтьева: (I - A)X = Y
    
    Особенности:
    - Поддержка множества правых частей (batch solve)
    - Многопоточное решение через joblib
    - Итерационные методы для больших матриц
    - Сравнение производительности
    """
    
    def __init__(self, I_minus_A: np.ndarray, tol: float = 1e-8, maxiter: int = 1000):
        """
        Args:
            I_minus_A: матрица (I - A) размером n x n
            tol: точность итерационных методов
            maxiter: максимальное число итераций
        """
        self.n = I_minus_A.shape[0]
        self.I_minus_A = I_minus_A.astype(np.float64)
        self.tol = tol
        self.maxiter = maxiter
        
        # Для больших матриц используем разреженное представление (экономия памяти)
        if self.n > 100:
            self.A_sparse = csr_matrix(I_minus_A)
            self.use_sparse = True
        else:
            self.A_sparse = I_minus_A
            self.use_sparse = False
    
    def solve_direct(self, Y: np.ndarray) -> np.ndarray:
        """
        Прямой метод решения (использует многопоточный LAPACK)
        Самый быстрый для матриц среднего размера (n < 500)
        """
        return np.linalg.solve(self.I_minus_A, Y.astype(np.float64))
    
    def solve_iterative(self, Y: np.ndarray, method: str = 'bicgstab') -> np.ndarray:
        """
        Итерационный метод (для больших матриц n > 500)
        
        Args:
            Y: правая часть
            method: 'bicgstab' или 'gmres'
        """
        Y = Y.astype(np.float64)
        
        if method == 'bicgstab':
            x, info = bicgstab(self.A_sparse, Y, tol=self.tol, maxiter=self.maxiter)
        elif method == 'gmres':
            x, info = gmres(self.A_sparse, Y, tol=self.tol, maxiter=self.maxiter, restart=50)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        # Если итерационный метод не сошёлся, используем прямой
        if info != 0:
            print(f"⚠️ Итерационный метод не сошёлся (info={info}),改用 прямой метод")
            return self.solve_direct(Y)
        
        return x
    
    def solve_batch_direct(self, Y_matrix: np.ndarray) -> np.ndarray:
        """
        Пакетное решение для МАТРИЦЫ правых частей
        Это КЛЮЧЕВОЙ метод для ускорения многовариантных расчётов!
        
        np.linalg.solve автоматически использует многопоточный LAPACK,
        что даёт значительное ускорение при решении нескольких СЛАУ одновременно.
        
        Args:
            Y_matrix: матрица (n, k) — k правых частей
        
        Returns:
            матрица решений (n, k)
        """
        return np.linalg.solve(self.I_minus_A, Y_matrix.astype(np.float64))
    
    def solve_parallel_joblib(self, Y_list: List[np.ndarray], 
                              method: str = 'direct',
                              n_jobs: int = -1) -> List[np.ndarray]:
        """
        Параллельное решение для списка правых частей через joblib
        
        Args:
            Y_list: список векторов правых частей
            method: 'direct' или 'iterative'
            n_jobs: количество потоков (-1 = все доступные ядра)
        
        Returns:
            список решений X
        """
        def solve_one(Y):
            if method == 'direct':
                return self.solve_direct(Y)
            else:
                return self.solve_iterative(Y, method)
        
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(solve_one)(Y) for Y in Y_list
        )
        
        return results
    
    def solve_auto(self, Y: Union[np.ndarray, List[np.ndarray]], 
                   prefer_batch: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Автоматический выбор наиболее эффективного метода
        
        Args:
            Y: вектор, список векторов или матрица правых частей
            prefer_batch: предпочитать пакетное решение
        
        Returns:
            решение или список решений
        """
        # Случай 1: Один вектор
        if isinstance(Y, np.ndarray) and Y.ndim == 1:
            if self.n > 500:
                return self.solve_iterative(Y, 'bicgstab')
            else:
                return self.solve_direct(Y)
        
        # Случай 2: Матрица правых частей
        if isinstance(Y, np.ndarray) and Y.ndim == 2:
            k = Y.shape[1]
            if prefer_batch or k > 10:
                return self.solve_batch_direct(Y)
            else:
                Y_list = [Y[:, i] for i in range(k)]
                return self.solve_parallel_joblib(Y_list)
        
        # Случай 3: Список векторов
        if isinstance(Y, list):
            k = len(Y)
            if prefer_batch and k > 5:
                Y_matrix = np.column_stack(Y)
                X_matrix = self.solve_batch_direct(Y_matrix)
                return [X_matrix[:, i] for i in range(k)]
            else:
                return self.solve_parallel_joblib(Y)
        
        raise TypeError(f"Неподдерживаемый тип Y: {type(Y)}")
    
    def benchmark(self, n_rhs: int = 100, verbose: bool = True) -> Dict:
        """
        Полный бенчмарк всех методов решения
        
        Args:
            n_rhs: количество правых частей
            verbose: выводить ли результат
        
        Returns:
            словарь с результатами
        """
        # Генерируем случайные правые части
        np.random.seed(42)
        Y_list = [np.random.randn(self.n) for _ in range(n_rhs)]
        Y_matrix = np.column_stack(Y_list)
        
        results = {
            'matrix_size': self.n,
            'n_rhs': n_rhs,
            'methods': {}
        }
        
        # 1. Пакетное решение (batch direct)
        start = time.perf_counter()
        X_batch = self.solve_batch_direct(Y_matrix)
        batch_time = time.perf_counter() - start
        results['methods']['batch_direct'] = {
            'time': batch_time,
            'speed_elements_per_sec': (self.n * n_rhs) / batch_time
        }
        
        # 2. Последовательное решение (по одному)
        start = time.perf_counter()
        X_seq = [self.solve_direct(Y) for Y in Y_list]
        seq_time = time.perf_counter() - start
        results['methods']['sequential'] = {
            'time': seq_time,
            'speed_elements_per_sec': (self.n * n_rhs) / seq_time
        }
        
        # 3. Параллельное joblib
        start = time.perf_counter()
        X_par = self.solve_parallel_joblib(Y_list, method='direct', n_jobs=-1)
        par_time = time.perf_counter() - start
        results['methods']['parallel_joblib'] = {
            'time': par_time,
            'speed_elements_per_sec': (self.n * n_rhs) / par_time
        }
        
        # 4. Итерационный метод (только для больших матриц)
        if self.n > 100:
            start = time.perf_counter()
            X_iter = [self.solve_iterative(Y, 'bicgstab') for Y in Y_list[:10]]
            iter_time = time.perf_counter() - start
            results['methods']['iterative_bicgstab'] = {
                'time': iter_time,
                'speed_elements_per_sec': (self.n * 10) / iter_time,
                'note': 'только 10 правых частей'
            }
        
        # Вычисляем ускорения
        seq_time = results['methods']['sequential']['time']
        results['speedup'] = {
            'batch_vs_seq': seq_time / results['methods']['batch_direct']['time'],
            'parallel_vs_seq': seq_time / results['methods']['parallel_joblib']['time']
        }
        
        # Определяем лучший метод
        best_method = min(results['methods'].items(), 
                         key=lambda x: x[1]['time'] if 'time' in x[1] else float('inf'))
        results['best_method'] = best_method[0]
        results['best_method_time'] = best_method[1]['time']
        
        if verbose:
            self._print_benchmark_results(results)
        
        return results
    
    def _print_benchmark_results(self, results: Dict):
        """Красивый вывод результатов бенчмарка"""
        print("\n" + "=" * 70)
        print(f"📊 БЕНЧМАРК РЕШЕНИЯ СЛАУ")
        print(f"   Размер матрицы: {results['matrix_size']}×{results['matrix_size']}")
        print(f"   Правых частей: {results['n_rhs']}")
        print("=" * 70)
        
        print("\n⏱️ Время выполнения:")
        for method, data in results['methods'].items():
            if 'time' in data:
                print(f"   • {method:20s}: {data['time']:.4f} сек")
        
        print("\n⚡ Ускорение:")
        for method, speedup in results['speedup'].items():
            print(f"   • {method:20s}: {speedup:.2f}x")
        
        print(f"\n🏆 Рекомендуемый метод: {results['best_method']}")
        print("=" * 70)


# ===================== ФУНКЦИИ ДЛЯ ИНТЕГРАЦИИ С МОДЕЛЬЮ =====================

def solve_leontief_system(I_minus_A: np.ndarray, 
                          Y: Union[np.ndarray, List[np.ndarray]],
                          method: str = 'auto') -> Union[np.ndarray, List[np.ndarray]]:
    """
    Унифицированная функция решения системы (I - A)X = Y
    
    Args:
        I_minus_A: матрица (I - A)
        Y: вектор, список векторов или матрица правых частей
        method: 
            - 'auto': автоматический выбор
            - 'direct': прямой метод solve
            - 'batch': пакетное решение (рекомендуется для нескольких правых частей)
            - 'parallel': параллельное решение через joblib
            - 'iterative': итерационный метод
    
    Returns:
        решение X
    """
    solver = ParallelSLESolver(I_minus_A)
    
    if method == 'auto':
        return solver.solve_auto(Y)
    elif method == 'direct':
        if isinstance(Y, np.ndarray) and Y.ndim == 2:
            return solver.solve_batch_direct(Y)
        elif isinstance(Y, np.ndarray) and Y.ndim == 1:
            return solver.solve_direct(Y)
        else:
            return solver.solve_parallel_joblib(Y, method='direct')
    elif method == 'batch':
        if isinstance(Y, np.ndarray) and Y.ndim == 2:
            return solver.solve_batch_direct(Y)
        elif isinstance(Y, list):
            Y_matrix = np.column_stack(Y)
            return solver.solve_batch_direct(Y_matrix)
        else:
            return solver.solve_direct(Y)
    elif method == 'parallel':
        if isinstance(Y, list):
            return solver.solve_parallel_joblib(Y, method='direct')
        else:
            return solver.solve_direct(Y)
    elif method == 'iterative':
        if isinstance(Y, np.ndarray) and Y.ndim == 1:
            return solver.solve_iterative(Y)
        else:
            return solver.solve_parallel_joblib(Y, method='bicgstab')
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def compare_solution_methods_for_country(country_code: str, year: int, 
                                         source: str = 'eurostat') -> Dict:
    """
    Сравнение методов решения для конкретной страны и года
    """
    from data_loader import EurostatDataLoader
    from exiobase_loader import EXIOBASELoader
    
    # Загружаем данные
    if source == 'exiobase':
        loader = EXIOBASELoader(country_code, year)
    else:
        loader = EurostatDataLoader(country_code, year)
    
    data = loader.get_input_output_tables()
    
    # Получаем матрицу Z и вектор X
    Z = data['Z']
    X = data['X']
    
    # Фильтруем нулевые отрасли
    valid = X > 0
    Z = Z[valid][:, valid]
    X = X[valid]
    
    # Ограничиваем размер
    max_n = 64 if source == 'eurostat' else 200
    n = min(len(X), max_n)
    Z = Z[:n, :n]
    X = X[:n]
    
    # Строим матрицу A
    X_safe = np.where(X > 0, X, 1.0)
    A = Z / X_safe
    I_minus_A = np.eye(n) - A
    
    # Генерируем тестовые правые части (сценарии)
    np.random.seed(42)
    n_scenarios = 50
    Y_list = [np.random.randn(n) * 100 for _ in range(n_scenarios)]
    
    # Запускаем бенчмарк
    solver = ParallelSLESolver(I_minus_A)
    results = solver.benchmark(n_rhs=n_scenarios, verbose=False)
    
    # Добавляем информацию о стране
    results['country'] = country_code
    results['year'] = year
    results['source'] = source
    results['actual_n'] = n
    
    return results


if __name__ == "__main__":
    """Тестирование"""
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ПАРАЛЛЕЛЬНОГО РЕШАТЕЛЯ СЛАУ")
    print("=" * 70)
    
    # Тест 1: Маленькая матрица (как в Eurostat)
    print("\n📊 ТЕСТ 1: Маленькая матрица 64×64")
    np.random.seed(42)
    I_minus_A_small = np.eye(64) - np.random.rand(64, 64) * 0.01
    solver_small = ParallelSLESolver(I_minus_A_small)
    solver_small.benchmark(n_rhs=50)
    
    # Тест 2: Средняя матрица
    print("\n📊 ТЕСТ 2: Средняя матрица 200×200")
    I_minus_A_medium = np.eye(200) - np.random.rand(200, 200) * 0.005
    solver_medium = ParallelSLESolver(I_minus_A_medium)
    solver_medium.benchmark(n_rhs=100)
    
    # Тест 3: Пример использования в модели
    print("\n📊 ТЕСТ 3: Решение системы с несколькими правыми частями")
    n = 64
    I_minus_A = np.eye(n) - np.random.rand(n, n) * 0.01
    
    # Генерируем 100 сценариев (разные Y)
    Y_list = [np.random.randn(n) * 1000 for _ in range(100)]
    
    # Пакетное решение
    start = time.perf_counter()
    Y_matrix = np.column_stack(Y_list)
    X_batch = np.linalg.solve(I_minus_A, Y_matrix)
    batch_time = time.perf_counter() - start
    
    print(f"   Пакетное решение {len(Y_list)} правых частей: {batch_time:.4f} сек")
    print(f"   Размер решений: {X_batch.shape}")