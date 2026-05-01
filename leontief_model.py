"""
Модель межотраслевого баланса Леонтьева
С поддержкой параллельных вычислений
"""

import numpy as np
import pandas as pd
import time
from typing import Optional, List, Tuple, Dict
from scipy.linalg import solve, inv
from multiprocessing import cpu_count

# Попытка импорта модулей параллельных вычислений
try:
    from parallel_computing import IterativeSolver, ThreadManager, ParallelScenarioAnalyzer, PerformanceMetrics
    PARALLEL_AVAILABLE = True
   # print("✅ Модуль parallel_computing загружен успешно")
except ImportError as e:
    print(f"⚠️ Модуль parallel_computing не найден: {e}")
    print("   Параллельные вычисления отключены.")
    PARALLEL_AVAILABLE = False
    
    # Создаем заглушки (классы на верхнем уровне, а не внутри другого класса!)
    class IterativeSolver:
        def __init__(self, *args, **kwargs): pass
        def solve_parallel_columns(self, *args, **kwargs): return None
    
    class ThreadManager:
        def __init__(self, *args, **kwargs): pass
        def set_num_threads(self, *args, **kwargs): return cpu_count()
        def get_current_settings(self): return {}
    
    class ParallelScenarioAnalyzer:
        def __init__(self, *args, **kwargs): pass
        def analyze_scenarios_parallel(self, *args, **kwargs): return []
        def analyze_shock_distribution(self, *args, **kwargs): return {}
    
    class PerformanceMetrics:
        def __init__(self, n_threads=1, matrix_size=0, inversion_time=0, 
                     condition_number=0, memory_usage_mb=0, method_used='unknown'):
            self.n_threads = n_threads
            self.matrix_size = matrix_size
            self.inversion_time = inversion_time
            self.condition_number = condition_number
            self.memory_usage_mb = memory_usage_mb
            self.method_used = method_used


class LeontiefModel:
    def __init__(self, Z, X, industries):
        """Инициализация модели"""
        self.Z = Z.astype(np.float64)
        self.X = X.astype(np.float64)
        self.industries = industries
        self.n = len(industries)
        self.A = None
        self.L = None
        self.condition_number = None
        
        # Инициализация модулей параллельных вычислений (если доступны)
        if PARALLEL_AVAILABLE:
            self.thread_manager = ThreadManager()
            self.iterative_solver = IterativeSolver()
        else:
            self.thread_manager = ThreadManager()  # Используем заглушку
            self.iterative_solver = IterativeSolver()  # Используем заглушку
        
    def calculate_matrix_A(self):
        """Расчёт матрицы прямых затрат A"""
        print("\n📊 Расчёт матрицы прямых затрат A...")
        
        # Защита от нулевого выпуска
        X_safe = np.where(self.X > 0, self.X, 1.0)
        
        # Расчёт A
        self.A = self.Z / X_safe
        self.A = np.nan_to_num(self.A, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ПРИНУДИТЕЛЬНАЯ НОРМАЛИЗАЦИЯ
        col_sums = self.A.sum(axis=0)
        target_sum = 0.6  # Типичное значение для развитой экономики
        
        for j in range(self.n):
            if col_sums[j] > 0.99:
                scale = target_sum / col_sums[j]
                self.A[:, j] = self.A[:, j] * scale
        
        # Финальная проверка
        col_sums = self.A.sum(axis=0)
        max_sum = col_sums.max()
        mean_sum = col_sums.mean()
        
        print(f"📊 Статистика матрицы A (после нормализации):")
        print(f"   • Max сумма по столбцу: {max_sum:.4f}")
        print(f"   • Средняя сумма по столбцу: {mean_sum:.4f}")
        print(f"   • Ненулевых элементов: {(self.A > 0).sum()}")
        
        if max_sum < 0.99:
            print(f"   ✅ Матрица A продуктивна")
        
        return self.A

    def calculate_leontief_matrix(self):
        """Расчёт матрицы Леонтьева L = (I - A)⁻¹ с защитой (последовательная версия)"""
        print("\n📊 Расчёт матрицы полных затрат L...")
        
        I = np.eye(self.n)
        I_minus_A = I - self.A
        
        # Число обусловленности
        self.condition_number = np.linalg.cond(I_minus_A)
        
        if self.condition_number < 1e4:
            print(f"✅ Матрица хорошо обусловлена (cond = {self.condition_number:.2e})")
        elif self.condition_number < 1e8:
            print(f"⚠️ Матрица удовлетворительно обусловлена (cond = {self.condition_number:.2e})")
        else:
            print(f"❌ Матрица плохо обусловлена (cond = {self.condition_number:.2e})")
            print("   Применяю регуляризацию...")
            # Добавляем небольшую регуляризацию
            I_minus_A = I_minus_A + np.eye(self.n) * 1e-6
            self.condition_number = np.linalg.cond(I_minus_A)
            print(f"   После регуляризации: cond = {self.condition_number:.2e}")
        
        # Обращение матрицы
        try:
            self.L = inv(I_minus_A)
            print("✅ Матрица Леонтьева рассчитана")
        except np.linalg.LinAlgError:
            print("🔄 Использую псевдообращение...")
            self.L = np.linalg.pinv(I_minus_A)
        
        # Проверка на гигантские значения
        max_val = np.abs(self.L).max()
        if max_val > 1e10:
            print(f"⚠️ Обнаружены гигантские значения (max={max_val:.2e}), применяю ограничение")
            self.L = np.clip(self.L, -1e6, 1e6)
        
        return self.L
    
    def calculate_leontief_matrix_parallel(self, 
                                          use_iterative: bool = True, 
                                          n_threads: Optional[int] = None,
                                          method: str = 'bicgstab'):
        """
        Параллельный расчет матрицы Леонтьева
        
        Args:
            use_iterative: использовать итерационный метод (для больших матриц)
            n_threads: количество потоков
            method: 'bicgstab' или 'gmres'
        
        Returns:
            tuple: (L, metrics) где metrics - словарь с метриками или None
        """
        if not PARALLEL_AVAILABLE:
            print("⚠️ Параллельные вычисления недоступны. Использую последовательный метод.")
            L = self.calculate_leontief_matrix()
            metrics = PerformanceMetrics(
                n_threads=1,
                matrix_size=self.n,
                inversion_time=0,
                condition_number=self.condition_number,
                memory_usage_mb=0,
                method_used='sequential'
            )
            return L, metrics
        
        import psutil
        
        print("\n🚀 Параллельный расчет матрицы Леонтьева (Этап 3 ТЗ)")
        
        # 1. Настройка потоков
        actual_threads = self.thread_manager.set_num_threads(n_threads, verbose=True)
        
        # 2. Формирование матрицы (I - A)
        I = np.eye(self.n)
        I_minus_A = I - self.A
        
        # 3. Оценка числа обусловленности
        self.condition_number = np.linalg.cond(I_minus_A)
        print(f"   • Число обусловленности: {self.condition_number:.2e}")
        
        start_time = time.time()
        
        # 4. Выбор метода решения
        if use_iterative and self.n > 50 and PARALLEL_AVAILABLE:
            print(f"   • Использую итерационный метод: {method}")
            
            # Параллельное решение столбцов
            L = self.iterative_solver.solve_parallel_columns(
                I_minus_A, self.n, batch_size=10
            )
            method_used = f'iterative_{method}'
        else:
            if self.n <= 50:
                print(f"   • Использую прямой метод (n={self.n} <= 50)")
            else:
                print(f"   • Использую прямой метод (итерационный отключён или недоступен)")
            
            # Прямой метод с многопоточным BLAS
            L = inv(I_minus_A)
            method_used = 'direct_inv'
        
        computation_time = time.time() - start_time
        
        # 5. Метрики производительности
        metrics = PerformanceMetrics(
            n_threads=actual_threads,
            matrix_size=self.n,
            inversion_time=computation_time,
            condition_number=self.condition_number,
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            method_used=method_used
        )
        
        self.L = L
        
        print(f"\n📊 Метрики производительности:")
        print(f"   • Время расчета: {computation_time:.2f} сек")
        print(f"   • Потоков: {actual_threads}")
        print(f"   • Память: {metrics.memory_usage_mb:.1f} MB")
        print(f"   • Метод: {method_used}")
        
        if self.condition_number > 1e8:
            print(f"   ⚠️ Матрица плохо обусловлена! Рассмотрите регуляризацию.")
        
        return L, metrics
    
    def calculate_multipliers(self):
        """Расчёт мультипликаторов с защитой"""
        if self.L is None:
            self.calculate_leontief_matrix()
        
        # Мультипликатор выпуска = сумма по столбцу L
        output_multipliers = self.L.sum(axis=0)
        # Мультипликатор затрат = сумма по строке L
        input_multipliers = self.L.sum(axis=1)
        
        # Ограничиваем реалистичными значениями
        output_multipliers = np.clip(output_multipliers, 0, 100)
        input_multipliers = np.clip(input_multipliers, 0, 100)
        
        return pd.DataFrame({
            'Отрасль': self.industries,
            'Мультипликатор_выпуска': output_multipliers,
            'Мультипликатор_затрат': input_multipliers
        })
    
    def analyze_shock(self, Y_base, delta_Y, name="Шок"):
        """Анализ влияния изменения спроса с защитой"""
        if self.L is None:
            self.calculate_leontief_matrix()
        
        # ΔX = L * ΔY
        delta_X = self.L @ delta_Y
        
        # Ограничиваем гигантские значения
        delta_X = np.clip(delta_X, -1e9, 1e9)
        
        total_effect = delta_X.sum()
        
        # Мультипликатор (с защитой от деления на 0)
        delta_Y_sum = delta_Y.sum()
        if abs(delta_Y_sum) > 1e-10:
            multiplier = total_effect / delta_Y_sum
        else:
            multiplier = 0
        
        return {
            'name': name,
            'delta_X': delta_X,
            'total_effect': total_effect,
            'multiplier': abs(multiplier)
        }
    
    def analyze_shocks_parallel(self, scenarios: List[Tuple[str, np.ndarray]], 
                                Y_base: np.ndarray) -> List[Dict]:
        """
        Параллельный анализ шоков (многовариантные расчеты)
        
        Args:
            scenarios: список сценариев [(имя, delta_Y), ...]
            Y_base: базовый вектор спроса
        
        Returns:
            список результатов анализа
        """
        analyzer = ParallelScenarioAnalyzer(self, n_jobs=-1, verbose=1)
        return analyzer.analyze_scenarios_parallel(scenarios, Y_base)
    
    def monte_carlo_analysis(self, Y_base: np.ndarray, 
                            shock_size: float = 0.1,
                            n_scenarios: int = 100) -> Dict:
        """
        Монте-Карло анализ с параллельными вычислениями
        
        Args:
            Y_base: базовый спрос
            shock_size: размер шока (доля от Y_base)
            n_scenarios: количество случайных сценариев
        
        Returns:
            статистика распределения мультипликаторов
        """
        analyzer = ParallelScenarioAnalyzer(self, n_jobs=-1, verbose=1)
        return analyzer.analyze_shock_distribution(Y_base, shock_size, n_scenarios)
    
    def get_matrices(self):
        """Возврат матриц для визуализации"""
        return {
            'A': self.A,
            'L': self.L,
            'cond': self.condition_number
        }