"""
Модуль параллельных вычислений для модели Леонтьева
Полная версия для Этапа 3 ТЗ
"""

import os
import numpy as np
from multiprocessing import cpu_count
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    n_threads: int
    matrix_size: int
    inversion_time: float
    condition_number: float
    memory_usage_mb: float
    method_used: str


class ThreadManager:
    """Управление потоками для MKL/OpenBLAS"""
    
    def __init__(self):
        self.original_env = {}
        
    def set_num_threads(self, n_threads: Optional[int] = None, verbose: bool = True) -> int:
        """Установка количества потоков"""
        if n_threads is None:
            n_threads = cpu_count()
        else:
            n_threads = min(n_threads, cpu_count())
        
        # Сохраняем оригинальные значения
        for env_var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
            self.original_env[env_var] = os.environ.get(env_var, None)
            os.environ[env_var] = str(n_threads)
        
        if verbose:
            print(f"🔧 Управление потоками:")
            print(f"   • CPU cores: {cpu_count()}")
            print(f"   • Используется потоков: {n_threads}")
            print(f"   • MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
        
        return n_threads
    
    def reset_threads(self):
        """Сброс настроек потоков"""
        for env_var, original_value in self.original_env.items():
            if original_value is None:
                if env_var in os.environ:
                    del os.environ[env_var]
            else:
                os.environ[env_var] = original_value
    
    def get_current_threads(self) -> int:
        """Получение текущего количества потоков"""
        return int(os.environ.get('MKL_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', cpu_count())))


class IterativeSolver:
    """Итерационные методы решения СЛАУ с параллельной реализацией"""
    
    def __init__(self, tol: float = 1e-8, maxiter: int = 1000):
        self.tol = tol
        self.maxiter = maxiter
    
    def set_tolerance(self, tol: float):
        """Установка точности"""
        self.tol = tol
        print(f"   • Точность установлена: {self.tol}")
    
    def set_maxiter(self, maxiter: int):
        """Установка максимального числа итераций"""
        self.maxiter = maxiter
        print(f"   • Макс. итераций: {self.maxiter}")
    
    def get_settings(self) -> Dict:
        """Получение текущих настроек"""
        return {
            'tolerance': self.tol,
            'maxiter': self.maxiter
        }
    
    def solve_parallel_columns(self, A, n_cols: int, batch_size: int = 10, method: str = 'bicgstab'):
        """
        Параллельное решение для нескольких правых частей
        
        Args:
            A: матрица (I - A)
            n_cols: количество столбцов
            batch_size: размер батча для параллельной обработки
            method: 'bicgstab' или 'gmres'
        """
        try:
            from joblib import Parallel, delayed
            from scipy.sparse.linalg import bicgstab, gmres
            from scipy.sparse import csr_matrix
            from scipy.linalg import solve
            
            # Конвертируем в разреженную матрицу для больших размеров
            if n_cols > 50:
                A_sparse = csr_matrix(A)
                print(f"   • Использую разреженное представление матрицы")
            else:
                A_sparse = A
            
            def solve_column(j):
                """Решение для одного столбца"""
                e_j = np.zeros(n_cols)
                e_j[j] = 1.0
                
                try:
                    if method == 'bicgstab':
                        x, info = bicgstab(A_sparse, e_j, tol=self.tol, maxiter=self.maxiter)
                    else:
                        x, info = gmres(A_sparse, e_j, tol=self.tol, maxiter=self.maxiter, restart=50)
                    
                    if info == 0:
                        return x
                    else:
                        # Fallback на прямой метод
                        return solve(A, e_j)
                except:
                    return solve(A, e_j)
            
            print(f"📊 Параллельное решение {n_cols} столбцов...")
            print(f"   • Потоков: {cpu_count()}")
            print(f"   • Метод: {method}")
            print(f"   • Точность: {self.tol}")
            print(f"   • Макс. итераций: {self.maxiter}")
            
            # Параллельное выполнение
            results = Parallel(n_jobs=-1, backend='loky', verbose=0)(
                delayed(solve_column)(j) for j in range(n_cols)
            )
            
            L = np.column_stack(results)
            print(f"   ✅ Решено {n_cols} столбцов")
            return L
            
        except ImportError as e:
            print(f"   ⚠️ Ошибка импорта: {e}")
            print("   Использую последовательное решение...")
            from scipy.linalg import solve
            L = np.zeros((n_cols, n_cols))
            for j in range(n_cols):
                e_j = np.zeros(n_cols)
                e_j[j] = 1.0
                L[:, j] = solve(A, e_j)
                if (j + 1) % 10 == 0:
                    print(f"   • Решено {j + 1}/{n_cols} столбцов")
            return L
    
    def solve_column_sequential(self, A, n_cols: int, method: str = 'bicgstab'):
        """Последовательное решение (для сравнения)"""
        from scipy.linalg import solve
        from scipy.sparse.linalg import bicgstab, gmres
        from scipy.sparse import csr_matrix
        
        if n_cols > 50:
            A_sparse = csr_matrix(A)
        else:
            A_sparse = A
        
        L = np.zeros((n_cols, n_cols))
        for j in range(n_cols):
            e_j = np.zeros(n_cols)
            e_j[j] = 1.0
            
            try:
                if method == 'bicgstab':
                    x, info = bicgstab(A_sparse, e_j, tol=self.tol, maxiter=self.maxiter)
                else:
                    x, info = gmres(A_sparse, e_j, tol=self.tol, maxiter=self.maxiter)
                
                if info == 0:
                    L[:, j] = x
                else:
                    L[:, j] = solve(A, e_j)
            except:
                L[:, j] = solve(A, e_j)
            
            if (j + 1) % 10 == 0:
                print(f"   • Решено {j + 1}/{n_cols} столбцов")
        
        return L


class ParallelScenarioAnalyzer:
    """Параллельный анализ сценариев"""
    
    def __init__(self, model, n_jobs: int = -1, verbose: int = 1):
        self.model = model
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def analyze_scenarios_parallel(self, scenarios: List[Tuple[str, np.ndarray]], 
                                   Y_base: np.ndarray,
                                   use_parallel: bool = True) -> List[Dict]:
        """Параллельный анализ нескольких сценариев"""
        
        if not use_parallel:
            # Последовательный режим
            results = []
            for name, delta in scenarios:
                results.append(self.model.analyze_shock(Y_base, delta, name))
            return results
        
        try:
            from joblib import Parallel, delayed
            
            if self.verbose:
                print(f"\n🚀 Параллельный анализ {len(scenarios)} сценариев")
                print(f"   • Потоков: {self.n_jobs if self.n_jobs > 0 else cpu_count()}")
                print(f"   • Бэкенд: loky")
            
            def process_scenario(scenario):
                name, delta = scenario
                return self.model.analyze_shock(Y_base, delta, name)
            
            results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=0)(
                delayed(process_scenario)(scenario) for scenario in scenarios
            )
            
            if self.verbose:
                print(f"   ✅ Обработано {len(results)} сценариев")
                # Подсчет статистики
                multipliers = [r['multiplier'] for r in results]
                print(f"   • Средний мультипликатор: {np.mean(multipliers):.3f}")
                print(f"   • Мин/макс: {np.min(multipliers):.3f} / {np.max(multipliers):.3f}")
            
            return results
            
        except ImportError as e:
            if self.verbose:
                print(f"   ⚠️ joblib не установлен: {e}")
                print("   Использую последовательный анализ...")
            
            results = []
            for name, delta in scenarios:
                results.append(self.model.analyze_shock(Y_base, delta, name))
            return results
    
    def analyze_shock_distribution(self, Y_base: np.ndarray, 
                                   shock_size: float = 0.1,
                                   n_shocks: int = 100) -> Dict:
        """Монте-Карло анализ распределения шоков"""
        print(f"\n🎲 Монте-Карло анализ ({n_shocks} сценариев)...")
        
        # Генерируем случайные шоки
        np.random.seed(42)
        scenarios = []
        
        for i in range(n_shocks):
            delta = np.random.randn(len(Y_base)) * shock_size * np.abs(Y_base)
            delta = np.clip(delta, -np.abs(Y_base) * 0.5, np.abs(Y_base) * 0.5)
            scenarios.append((f"Random_{i+1}", delta))
        
        # Анализ
        results = self.analyze_scenarios_parallel(scenarios, Y_base)
        
        # Статистика
        multipliers = [r['multiplier'] for r in results]
        total_effects = [r['total_effect'] for r in results]
        
        stats = {
            'mean_multiplier': np.mean(multipliers),
            'std_multiplier': np.std(multipliers),
            'mean_total_effect': np.mean(total_effects),
            'std_total_effect': np.std(total_effects),
            'percentile_95_multiplier': np.percentile(multipliers, 95),
            'percentile_5_multiplier': np.percentile(multipliers, 5),
            'min_multiplier': np.min(multipliers),
            'max_multiplier': np.max(multipliers)
        }
        
        print(f"\n📊 Статистика по {n_shocks} сценариям:")
        print(f"   • Средний мультипликатор: {stats['mean_multiplier']:.3f} ± {stats['std_multiplier']:.3f}")
        print(f"   • 95й перцентиль: {stats['percentile_95_multiplier']:.3f}")
        print(f"   • 5й перцентиль: {stats['percentile_5_multiplier']:.3f}")
        print(f"   • Диапазон: [{stats['min_multiplier']:.3f}, {stats['max_multiplier']:.3f}]")
        
        return stats
    
    def benchmark_methods(self, Y_base: np.ndarray, n_scenarios: int = 20) -> Dict:
        """Сравнение производительности разных методов"""
        print("\n📊 Сравнение методов анализа сценариев...")
        
        # Генерируем тестовые сценарии
        np.random.seed(42)
        scenarios = []
        for i in range(n_scenarios):
            delta = np.random.randn(len(Y_base)) * 0.1 * np.abs(Y_base)
            delta = np.clip(delta, -np.abs(Y_base) * 0.3, np.abs(Y_base) * 0.3)
            scenarios.append((f"Test_{i}", delta))
        
        import time
        
        # Последовательный
        start = time.time()
        seq_results = []
        for name, delta in scenarios:
            seq_results.append(self.model.analyze_shock(Y_base, delta, name))
        seq_time = time.time() - start
        
        # Параллельный
        start = time.time()
        par_results = self.analyze_scenarios_parallel(scenarios, Y_base, use_parallel=True)
        par_time = time.time() - start
        
        return {
            'sequential_time': seq_time,
            'parallel_time': par_time,
            'speedup': seq_time / par_time if par_time > 0 else 0,
            'n_scenarios': n_scenarios
        }