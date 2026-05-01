#!/usr/bin/env python3
"""
Главный скрипт расчёта модели Леонтьева
С поддержкой параллельных вычислений (Этап 3 ТЗ)
Поддерживает Eurostat (64 отрасли) и EXIOBASE (200+ отраслей)
"""

import argparse
import numpy as np
import pandas as pd
import warnings
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from config import (
    DEFAULT_COUNTRY, DEFAULT_YEAR, OUTPUT_DIR, MAX_INDUSTRIES,
    ensure_directories, get_country_name, AVAILABLE_COUNTRIES, AVAILABLE_YEARS,
    EXIOBASE_COUNTRIES, EXIOBASE_YEARS, EXIOBASE_MAX_INDUSTRIES
)
from data_loader import EurostatDataLoader
from exiobase_loader import EXIOBASELoader
from leontief_model import LeontiefModel
from visualization import create_all_visualizations
from parallel_computing import ThreadManager, PerformanceMetrics, IterativeSolver, ParallelScenarioAnalyzer


warnings.filterwarnings('ignore')
ensure_directories()


def parse_args():
    """Парсинг аргументов командной строки с поддержкой параллельных вычислений и выбора источника"""
    parser = argparse.ArgumentParser(
        description='Leontief Input-Output Model with Parallel Computing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py                              # Франция, 2020, Eurostat (64 отрасли)
  python main.py --country DE --year 2015     # Германия, 2015, Eurostat
  python main.py --source exiobase            # EXIOBASE (200+ отраслей)
  python main.py --source exiobase --country US --year 2020  # США, EXIOBASE
  python main.py --iterative                  # Использовать итерационный метод
  python main.py --threads 8                  # Использовать 8 потоков
  python main.py --monte-carlo 100            # Монте-Карло анализ 100 сценариев
  python main.py --method gmres               # Использовать GMRES вместо BICGSTAB
        """
    )
    
    # Основные аргументы
    parser.add_argument('--country', '-c', type=str, default=DEFAULT_COUNTRY,
                        help=f'Код страны (по умолчанию: {DEFAULT_COUNTRY})')
    parser.add_argument('--year', '-y', type=int, default=DEFAULT_YEAR,
                        help=f'Год данных (по умолчанию: {DEFAULT_YEAR})')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Директория для сохранения результатов (по умолчанию: outputs/{country}_{year}_{source})')
    
    # Аргументы для выбора источника данных
    parser.add_argument('--source', '-s', type=str, default='eurostat',
                        choices=['eurostat', 'exiobase'],
                        help='Источник данных: eurostat (64 отрасли) или exiobase (200+ отраслей)')
    
    # Аргументы для параллельных вычислений (Этап 3 ТЗ)
    parser.add_argument('--iterative', action='store_true',
                        help='Использовать итерационный метод для больших матриц')
    parser.add_argument('--threads', '-t', type=int, default=None,
                        help='Количество потоков для вычислений (по умолчанию: все доступные)')
    parser.add_argument('--method', choices=['bicgstab', 'gmres'], default='bicgstab',
                        help='Итерационный метод (bicgstab/gmres), по умолчанию: bicgstab')
    parser.add_argument('--monte-carlo', type=int, default=0,
                        help='Количество сценариев для Монте-Карло анализа (0 = отключено)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Отключить параллельные вычисления (использовать последовательный режим)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Запустить бенчмарк производительности')
    
    return parser.parse_args()


def get_data_loader(source: str, country_code: str, year: int):
    """Возвращает загрузчик данных для указанного источника"""
    if source == 'exiobase':
        return EXIOBASELoader(country_code, year)
    else:
        return EurostatDataLoader(country_code, year)


def get_available_countries_for_source(source: str) -> dict:
    """Возвращает список стран для источника"""
    if source == 'exiobase':
        return EXIOBASE_COUNTRIES
    else:
        return AVAILABLE_COUNTRIES


def get_available_years_for_source(source: str) -> list:
    """Возвращает список лет для источника"""
    if source == 'exiobase':
        return EXIOBASE_YEARS
    else:
        return AVAILABLE_YEARS


def get_source_info(source: str) -> dict:
    """Возвращает информацию об источнике"""
    if source == 'exiobase':
        return {
            'name': 'EXIOBASE',
            'description': 'Глобальная экологически-ориентированная ММВ-таблица',
            'max_industries': EXIOBASE_MAX_INDUSTRIES,
            'icon': '🌍'
        }
    else:
        return {
            'name': 'Eurostat',
            'description': 'Официальные данные Евросоюза (NACE Rev.2)',
            'max_industries': MAX_INDUSTRIES,
            'icon': '🇪🇺'
        }


def find_industry(industries, keywords):
    """
    Поиск индекса отрасли по ключевым словам
    
    Args:
        industries: список названий отраслей
        keywords: список ключевых слов для поиска
    
    Returns:
        индекс отрасли или None
    """
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for i, ind in enumerate(industries):
            ind_lower = ind.lower()
            if keyword.upper() in ind:
                return i
            if keyword_lower in ind_lower:
                return i
    return None


def print_available_industries(industries, keyword=None):
    """Вывод доступных отраслей для отладки"""
    print("\n📋 Доступные отрасли:")
    if keyword:
        print(f"   (поиск по ключевому слову: '{keyword}')")
        matching = [ind for ind in industries if keyword.lower() in ind.lower()]
        for ind in matching[:10]:
            print(f"   • {ind}")
        if not matching:
            print("   ❌ Ничего не найдено")
    else:
        for i, ind in enumerate(industries[:10]):
            print(f"   • {ind}")
        if len(industries) > 10:
            print(f"   ... и ещё {len(industries) - 10} отраслей")


def prepare_scenarios(industries, Y_base, X, n):
    """
    Подготовка сценариев для анализа
    
    Returns:
        List[Tuple[str, np.ndarray]]: список сценариев (имя, delta_Y)
    """
    print("\n   🔍 Проверка наличия отраслей для сценариев:")
    
    scenarios = []
    
    # Сценарий 1: Рост обрабатывающей промышленности (+10%)
    man_keywords = ['manufacturing', 'machinery', 'metal', 'chemical', 'motor', 'food', 
                    'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 
                    'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 
                    'C28', 'C29', 'C30', 'C31', 'C32', 'C33', 'Manufacturing']
    man_idx = find_industry(industries, man_keywords)
    if man_idx is not None:
        print(f"   ✅ Промышленность: {industries[man_idx][:50]}")
        delta = np.zeros(n)
        shock_value = min(Y_base[man_idx] * 0.1, X[man_idx] * 0.1)
        if shock_value > 0:
            delta[man_idx] = abs(shock_value)
            scenarios.append(('Рост_промышленности_10%', delta))
    else:
        print(f"   ❌ Промышленность: не найдена")
    
    # Сценарий 2: Рост строительства (+15%)
    const_keywords = ['construction', 'F', 'Construction']
    const_idx = find_industry(industries, const_keywords)
    if const_idx is not None:
        print(f"   ✅ Строительство: {industries[const_idx][:50]}")
        delta = np.zeros(n)
        shock_value = min(Y_base[const_idx] * 0.15, X[const_idx] * 0.15)
        if shock_value > 0:
            delta[const_idx] = abs(shock_value)
            scenarios.append(('Рост_строительства_15%', delta))
    else:
        print(f"   ❌ Строительство: не найдено")
    
    # Сценарий 3: Спад транспорта (-10%)
    trans_keywords = ['transport', 'logistic', 'H49', 'H50', 'H51', 'H52', 'H53', 'Transport']
    trans_idx = find_industry(industries, trans_keywords)
    if trans_idx is not None:
        print(f"   ✅ Транспорт: {industries[trans_idx][:50]}")
        delta = np.zeros(n)
        shock_value = min(abs(Y_base[trans_idx]) * 0.1, X[trans_idx] * 0.1)
        if shock_value > 0:
            delta[trans_idx] = -abs(shock_value)
            scenarios.append(('Спад_транспорта_10%', delta))
    else:
        print(f"   ❌ Транспорт: не найден")
    
    # Сценарий 4: Комплексный рост
    if man_idx is not None and const_idx is not None:
        delta = np.zeros(n)
        val1 = min(Y_base[man_idx] * 0.05, X[man_idx] * 0.05)
        val2 = min(Y_base[const_idx] * 0.05, X[const_idx] * 0.05)
        if val1 > 0 and val2 > 0:
            delta[man_idx] = abs(val1)
            delta[const_idx] = abs(val2)
            scenarios.append(('Комплексный_рост_5%x2', delta))
    
    # Сценарий 5: Рост IT-сектора (+20%)
    it_keywords = ['IT', 'computer', 'software', 'information', 'telecommunication', 
                   'publishing', 'J58', 'J59', 'J60', 'J61', 'J62', 'J63', 'IT', 'Information']
    it_idx = find_industry(industries, it_keywords)
    if it_idx is not None:
        print(f"   ✅ IT-сектор: {industries[it_idx][:50]}")
        delta = np.zeros(n)
        shock_value = min(Y_base[it_idx] * 0.2, X[it_idx] * 0.2)
        if shock_value > 0:
            delta[it_idx] = abs(shock_value)
            scenarios.append(('Рост_IT_20%', delta))
    else:
        print(f"   ❌ IT-сектор: не найден")
    
    # Сценарий 6: Рост здравоохранения (+10%)
    health_keywords = ['health', 'human health', 'medical', 'pharma', 'Q86', 'Q87', 'Q88', 
                       'social work', 'residential care', 'hospital', 'Health']
    health_idx = find_industry(industries, health_keywords)
    if health_idx is not None:
        print(f"   ✅ Здравоохранение: {industries[health_idx][:50]}")
        delta = np.zeros(n)
        shock_value = min(Y_base[health_idx] * 0.1, X[health_idx] * 0.1)
        if shock_value > 0:
            delta[health_idx] = abs(shock_value)
            scenarios.append(('Рост_здравоохранения_10%', delta))
    else:
        print(f"   ❌ Здравоохранение: не найдено")
        print("\n   📋 Похожие отрасли в данных:")
        for ind in industries:
            if any(kw in ind.lower() for kw in ['care', 'social', 'human', 'health']):
                print(f"      • {ind}")
    
    return scenarios


def run_benchmark(model, Y_base, n_scenarios: int = 50):
    """
    Запуск бенчмарка производительности
    
    Args:
        model: экземпляр LeontiefModel
        Y_base: базовый спрос
        n_scenarios: количество тестовых сценариев
    """
    print("\n" + "=" * 70)
    print("📊 БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 70)
    
    results = {}
    
    # Тест 1: Разные методы решения
    methods = ['direct', 'iterative_bicgstab', 'iterative_gmres']
    
    for method in methods:
        print(f"\n🔬 Тестирование метода: {method}")
        
        start_time = time.time()
        
        if method == 'direct':
            _, metrics = model.calculate_leontief_matrix_parallel(
                use_iterative=False, n_threads=args.threads
            )
        elif method == 'iterative_bicgstab':
            _, metrics = model.calculate_leontief_matrix_parallel(
                use_iterative=True, n_threads=args.threads, method='bicgstab'
            )
        else:  # iterative_gmres
            _, metrics = model.calculate_leontief_matrix_parallel(
                use_iterative=True, n_threads=args.threads, method='gmres'
            )
        
        results[method] = {
            'time': metrics.inversion_time,
            'memory': metrics.memory_usage_mb,
            'condition': metrics.condition_number
        }
        
        print(f"   Время: {metrics.inversion_time:.2f} сек")
        print(f"   Память: {metrics.memory_usage_mb:.1f} MB")
    
    # Тест 2: Параллельный анализ сценариев
    print(f"\n🔬 Тестирование параллельного анализа {n_scenarios} сценариев...")
    
    # Генерируем случайные сценарии
    test_scenarios = []
    np.random.seed(42)
    for i in range(n_scenarios):
        delta = np.random.randn(len(Y_base)) * 0.1 * np.abs(Y_base)
        delta = np.clip(delta, -np.abs(Y_base) * 0.5, np.abs(Y_base) * 0.5)
        test_scenarios.append((f"Test_{i}", delta))
    
    start_time = time.time()
    shock_results = model.analyze_shocks_parallel(test_scenarios, Y_base)
    parallel_time = time.time() - start_time
    
    results['parallel_scenarios'] = {
        'n_scenarios': n_scenarios,
        'time': parallel_time,
        'time_per_scenario': parallel_time / n_scenarios
    }
    
    print(f"   Общее время: {parallel_time:.2f} сек")
    print(f"   Время на сценарий: {parallel_time/n_scenarios:.3f} сек")
    
    # Сохранение результатов бенчмарка
    benchmark_file = Path("outputs/benchmark_results.json")
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Результаты бенчмарка сохранены в {benchmark_file}")
    
    # Вывод рекомендаций
    print("\n📈 РЕКОМЕНДАЦИИ:")
    best_method = min(results.keys(), key=lambda x: results[x]['time'] if isinstance(results[x], dict) and 'time' in results[x] else float('inf'))
    if best_method in results:
        print(f"   • Самый быстрый метод: {best_method} ({results[best_method]['time']:.2f} сек)")
    
    if model.condition_number > 1e8:
        print(f"   • ⚠️ Матрица плохо обусловлена. Рекомендуется использовать итерационные методы.")
    elif model.condition_number > 1e4:
        print(f"   • ⚠️ Матрица удовлетворительно обусловлена. Прямой метод может работать.")
    else:
        print(f"   • ✅ Матрица хорошо обусловлена. Любой метод будет работать.")
    
    return results


def main():
    global args
    args = parse_args()
    
    # Получаем информацию об источнике данных
    source_info = get_source_info(args.source)
    
    # Валидация страны и года для выбранного источника
    available_countries = get_available_countries_for_source(args.source)
    available_years = get_available_years_for_source(args.source)
    
    if args.country not in available_countries:
        print(f"⚠️ Страна {args.country} не доступна для {source_info['name']}")
        print(f"   Доступные страны: {', '.join(available_countries.keys())}")
        args.country = list(available_countries.keys())[0]
        print(f"   Использую {args.country}")
    
    if args.year not in available_years:
        print(f"⚠️ Год {args.year} не доступен для {source_info['name']}")
        print(f"   Доступные годы: {available_years}")
        args.year = available_years[-1]
        print(f"   Использую {args.year}")
    
    # Создаем директорию для конкретной страны, года и источника
    if args.output_dir:
        country_output_dir = Path(args.output_dir)
    else:
        country_output_dir = Path(f"outputs/{args.country}_{args.year}_{args.source}")
    
    country_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"🏭 МОДЕЛЬ МЕЖОТРАСЛЕВОГО БАЛАНСА ЛЕОНТЬЕВА")
    print(f"📊 {source_info['icon']} {source_info['name']} - {source_info['description']}")
    print("=" * 70)
    
    # Настройка управления потоками
    thread_manager = ThreadManager()
    if args.threads:
        actual_threads = thread_manager.set_num_threads(args.threads, verbose=True)
    else:
        actual_threads = thread_manager.set_num_threads(verbose=True)
    
    # ===== 1. ЗАГРУЗКА ДАННЫХ =====
    print(f"\n📥 ЭТАП 1: Загрузка данных из {source_info['name']}")
    country_name = available_countries.get(args.country, args.country)
    print(f"   Страна: {country_name} ({args.country})")
    print(f"   Год: {args.year}")
    print(f"   Директория: {country_output_dir}")
    print(f"   Источник: {args.source}")
    
    loader = get_data_loader(args.source, args.country, args.year)
    data = loader.get_input_output_tables()
    
    # Проверка и очистка данных
    Z = data['Z']
    X = data['X']
    Y = data['Y']
    industries = data['industries']
    
    # Убираем строки/столбцы с нулевым выпуском
    valid_mask = X > 0
    Z = Z[valid_mask][:, valid_mask]
    X = X[valid_mask]
    Y = Y[valid_mask]
    industries = [ind for i, ind in enumerate(industries) if valid_mask[i]]
    
    # Ограничиваем размер
    max_n = source_info['max_industries']
    n = min(len(industries), max_n)
    Z = Z[:n, :n]
    X = X[:n]
    Y = Y[:n]
    industries = industries[:n]
    
    # Нормализация данных
    scale_factor = 1.0
    if X.mean() > 1e6:
        scale_factor = 1e6
        Z = Z / scale_factor
        X = X / scale_factor
        Y = Y / scale_factor
    
    print(f"\n📊 Загружено:")
    print(f"   • Источник: {source_info['name']} ({args.source})")
    print(f"   • Отраслей: {n}")
    print(f"   • Размер матрицы Z: {Z.shape}")
    print(f"   • Суммарный выпуск: {X.sum():.1f} млн €")
    print(f"   • Средний выпуск на отрасль: {X.mean():.1f} млн €")
    
    # Обновляем данные
    data['Z'] = Z
    data['X'] = X
    data['Y'] = Y
    data['industries'] = industries
    data['n'] = n
    
    # ===== 2. РАСЧЁТ МАТРИЦЫ A =====
    print("\n📐 ЭТАП 2: Матрица прямых затрат A")
    model = LeontiefModel(data['Z'], data['X'], data['industries'])
    A = model.calculate_matrix_A()
    
    # Сохранение
    pd.DataFrame(A, index=data['industries'], columns=data['industries']).to_csv(
        country_output_dir / "matrix_A.csv"
    )
    print(f"   ✅ Сохранено: {country_output_dir}/matrix_A.csv")
    
    # ===== 3. МАТРИЦА L (ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ) =====
    print("\n📐 ЭТАП 3: Матрица полных затрат L (Параллельные вычисления)")
    
    # Выбор метода на основе аргументов и размера матрицы
    use_iterative = args.iterative or (n > 50 and not args.no_parallel)
    
    if args.no_parallel:
        print("   ℹ️ Параллельные вычисления отключены (--no-parallel)")
        print("   Использую последовательный метод...")
        L = model.calculate_leontief_matrix()
        metrics = PerformanceMetrics(
            n_threads=1,
            matrix_size=n,
            inversion_time=0,
            condition_number=model.condition_number,
            memory_usage_mb=0,
            method_used='sequential'
        )
    else:
        # Расчет с параллельными методами
        L, metrics = model.calculate_leontief_matrix_parallel(
            use_iterative=use_iterative,
            n_threads=args.threads,
            method=args.method
        )
    
    # Сохранение матрицы L
    pd.DataFrame(L, index=data['industries'], columns=data['industries']).to_csv(
        country_output_dir / "matrix_L.csv"
    )
    print(f"   ✅ Сохранено: {country_output_dir}/matrix_L.csv")
    
    # Сохранение метрик производительности
    with open(country_output_dir / "performance_metrics.json", 'w') as f:
        json.dump({
            'source': args.source,
            'n_threads': metrics.n_threads,
            'matrix_size': metrics.matrix_size,
            'inversion_time': metrics.inversion_time,
            'condition_number': metrics.condition_number,
            'memory_usage_mb': metrics.memory_usage_mb,
            'method_used': metrics.method_used,
            'use_iterative': use_iterative,
            'iterative_method': args.method if use_iterative else None
        }, f, indent=2)
    
    # ===== 4. МУЛЬТИПЛИКАТОРЫ =====
    print("\n📈 ЭТАП 4: Расчёт мультипликаторов")
    multipliers_df = model.calculate_multipliers()
    multipliers_df.to_csv(country_output_dir / "multipliers.csv", index=False)
    print(f"   ✅ Сохранено: {country_output_dir}/multipliers.csv")
    
    # Топ-5
    print("\n   🔝 Топ-5 отраслей по влиянию:")
    top5 = multipliers_df.nlargest(5, 'Мультипликатор_выпуска')
    for _, row in top5.iterrows():
        name = row['Отрасль'][:45]
        mult = row['Мультипликатор_выпуска']
        print(f"      • {name}: {mult:.3f}")
    
    # ===== 5. СЦЕНАРНЫЙ АНАЛИЗ (ПАРАЛЛЕЛЬНЫЙ) =====
    print("\n🎯 ЭТАП 5: Сценарный анализ (параллельный)")
    Y_base = data['Y'].copy()
    n = data['n']
    
    # Подготовка сценариев
    scenarios = prepare_scenarios(industries, Y_base, X, n)
    
    shock_results = []
    if scenarios:
        print(f"\n📊 Подготовлено {len(scenarios)} сценариев для анализа")
        
        if args.no_parallel:
            # Последовательный анализ
            print("   Использую последовательный анализ...")
            for name, delta in scenarios:
                result = model.analyze_shock(Y_base, delta, name)
                
                if (abs(result['total_effect']) < 1e12 and 
                    abs(result['multiplier']) < 100 and 
                    abs(result['multiplier']) > 0.01 and
                    abs(result['total_effect']) > 0.01):
                    shock_results.append(result)
        else:
            # Параллельный анализ
            print(f"   Использую параллельный анализ с {actual_threads} потоками...")
            shock_results = model.analyze_shocks_parallel(scenarios, Y_base)
        
        # Сохранение и вывод результатов
        for result in shock_results:
            df = pd.DataFrame({
                'Отрасль': data['industries'],
                'Изменение_выпуска': result['delta_X']
            })
            safe_name = result['name'].replace(' ', '_').replace('%', '').replace('×', 'x')
            df.to_csv(country_output_dir / f"shock_{safe_name}.csv", index=False)
            
            print(f"\n   📊 {result['name'].replace('_', ' ')}")
            print(f"      Общий эффект: {result['total_effect']:,.1f} млн €")
            print(f"      Мультипликатор: {result['multiplier']:.3f}")
    
    # ===== 5.5 МОНТЕ-КАРЛО АНАЛИЗ (опционально) =====
    if args.monte_carlo > 0 and not args.no_parallel:
        print("\n🎲 Монте-Карло анализ...")
        try:
            mc_stats = model.monte_carlo_analysis(
                Y_base, 
                shock_size=0.1, 
                n_scenarios=args.monte_carlo
            )
            
            with open(country_output_dir / "monte_carlo_stats.json", 'w') as f:
                json.dump(mc_stats, f, indent=2)
            
            print(f"\n   📊 Статистика по {args.monte_carlo} сценариям:")
            print(f"      • Средний мультипликатор: {mc_stats['mean_multiplier']:.3f} ± {mc_stats['std_multiplier']:.3f}")
            print(f"      • 95й перцентиль: {mc_stats['percentile_95_multiplier']:.3f}")
            print(f"      • 5й перцентиль: {mc_stats['percentile_5_multiplier']:.3f}")
        except Exception as e:
            print(f"   ⚠️ Ошибка Монте-Карло анализа: {e}")
    
    # ===== 6. ВИЗУАЛИЗАЦИЯ =====
    print("\n🎨 ЭТАП 6: Создание визуализаций")
    
    import visualization
    original_plots_dir = visualization.PLOTS_DIR
    visualization.PLOTS_DIR = country_output_dir / "plots"
    visualization.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    create_all_visualizations(model, multipliers_df, shock_results if shock_results else None)
    
    visualization.PLOTS_DIR = original_plots_dir
    
    # ===== БЕНЧМАРК (опционально) =====
    if args.benchmark and not args.no_parallel:
        run_benchmark(model, Y_base, n_scenarios=min(50, len(scenarios) if scenarios else 20))
    
    # ===== ИТОГИ =====
    print("\n" + "=" * 70)
    print("📊 РАСЧЁТ ЗАВЕРШЁН")
    print("=" * 70)
    print(f"\n   ✅ Модель успешно рассчитана для {country_name} ({args.year})!")
    print(f"   📊 Источник: {source_info['name']} ({args.source}) - {source_info['description']}")
    print(f"\n   📁 Результаты в папке: {country_output_dir}/")
    print(f"      • matrix_A.csv — прямые затраты")
    print(f"      • matrix_L.csv — полные затраты")
    print(f"      • multipliers.csv — мультипликаторы")
    print(f"      • shock_*.csv — сценарии")
    print(f"      • plots/ — визуализации")
    print(f"      • performance_metrics.json — метрики производительности")
    
    if args.monte_carlo > 0:
        print(f"      • monte_carlo_stats.json — статистика Монте-Карло")
    
    print(f"\n   🚀 Для запуска веб-интерфейса:")
    print(f"      streamlit run app.py")
    print(f"      (в интерфейсе выберите источник в боковой панели)")
    
    # Дополнительная информация о параллельных вычислениях
    if not args.no_parallel:
        print(f"\n   ⚡ Параллельные вычисления:")
        print(f"      • Потоков: {actual_threads}")
        print(f"      • Метод: {args.method if use_iterative else 'прямой'}")
        print(f"      • Итерационный: {'да' if use_iterative else 'нет'}")
        
        if use_iterative:
            print(f"      • Точность: 1e-8")
            print(f"      • Макс. итераций: 1000")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()