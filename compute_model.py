"""
Модуль для расчёта модели Леонтьева "на лету" из Streamlit
"""

import numpy as np
import pandas as pd
from pathlib import Path
from data_loader import EurostatDataLoader
from leontief_model import LeontiefModel
from config import MAX_INDUSTRIES
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_and_save(country_code: str, year: int, force_recompute: bool = False) -> bool:
    """
    Загружает данные из Eurostat, рассчитывает модель и сохраняет результаты
    
    Args:
        country_code: код страны (FR, DE, IT...)
        year: год (2010-2022)
        force_recompute: принудительный пересчёт
    
    Returns:
        bool: успешно ли выполнено
    """
    output_dir = Path(f"outputs/{country_code}_{year}")
    
    # Проверяем, есть ли уже результаты
    if not force_recompute and (output_dir / "matrix_A.csv").exists():
        logger.info(f"✅ Данные для {country_code} {year} уже существуют")
        return True
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Загрузка данных из Eurostat
        logger.info(f"📥 Загрузка данных для {country_code} {year}...")
        loader = EurostatDataLoader(country_code, year)
        data = loader.get_input_output_tables()
        
        Z = data['Z']
        X = data['X']
        industries = data['industries']
        
        # Очистка и нормализация
        valid_mask = X > 0
        Z = Z[valid_mask][:, valid_mask]
        X = X[valid_mask]
        industries = [ind for i, ind in enumerate(industries) if valid_mask[i]]
        
        n = min(len(industries), MAX_INDUSTRIES)
        Z = Z[:n, :n]
        X = X[:n]
        industries = industries[:n]
        
        # Нормализация масштаба
        if X.mean() > 1e6:
            scale = 1e6
            Z = Z / scale
            X = X / scale
        
        # 2. Создание модели
        model = LeontiefModel(Z, X, industries)
        
        # 3. Расчёт матрицы A
        logger.info("📐 Расчёт матрицы A...")
        A = model.calculate_matrix_A()
        
        # 4. Расчёт матрицы L
        logger.info("🔢 Расчёт матрицы L...")
        L, metrics = model.calculate_leontief_matrix_parallel(use_iterative=True)
        
        # 5. Расчёт мультипликаторов
        logger.info("📈 Расчёт мультипликаторов...")
        multipliers_df = model.calculate_multipliers()
        
        # 6. Сохранение результатов
        pd.DataFrame(A, index=industries, columns=industries).to_csv(output_dir / "matrix_A.csv")
        pd.DataFrame(L, index=industries, columns=industries).to_csv(output_dir / "matrix_L.csv")
        multipliers_df.to_csv(output_dir / "multipliers.csv", index=False)
        
        # 7. Базовые сценарии
        Y_base = data['Y'][:n]
        scenarios = prepare_basic_scenarios(industries, Y_base, X, n)
        
        for name, delta in scenarios:
            result = model.analyze_shock(Y_base, delta, name)
            df = pd.DataFrame({
                'Отрасль': industries,
                'Изменение_выпуска': result['delta_X']
            })
            safe_name = name.replace(' ', '_').replace('%', '').replace('×', 'x')
            df.to_csv(output_dir / f"shock_{safe_name}.csv", index=False)
        
        # 8. Сохраняем метрики
        import json
        with open(output_dir / "performance_metrics.json", 'w') as f:
            json.dump({
                'n_industries': n,
                'condition_number': float(metrics.condition_number) if metrics.condition_number else None,
                'method_used': metrics.method_used,
                'inversion_time': metrics.inversion_time,
                'memory_usage_mb': metrics.memory_usage_mb
            }, f, indent=2)
        
        logger.info(f"✅ Расчёт для {country_code} {year} завершён")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка расчёта для {country_code} {year}: {e}")
        import traceback
        traceback.print_exc()
        return False


def prepare_basic_scenarios(industries, Y_base, X, n):
    """Подготовка базовых сценариев для анализа"""
    scenarios = []
    
    def find_industry(keywords):
        for keyword in keywords:
            for i, ind in enumerate(industries):
                if keyword.upper() in ind or keyword.lower() in ind.lower():
                    return i
        return None
    
    # Сценарий 1: Промышленность
    man_idx = find_industry(['manufacturing', 'machinery', 'metal', 'chemical', 'C10', 'C11'])
    if man_idx is not None:
        delta = np.zeros(n)
        shock_value = min(Y_base[man_idx] * 0.1, X[man_idx] * 0.1)
        if shock_value > 0:
            delta[man_idx] = abs(shock_value)
            scenarios.append(('Рост_промышленности_10%', delta))
    
    # Сценарий 2: Строительство
    const_idx = find_industry(['construction', 'F'])
    if const_idx is not None:
        delta = np.zeros(n)
        shock_value = min(Y_base[const_idx] * 0.15, X[const_idx] * 0.15)
        if shock_value > 0:
            delta[const_idx] = abs(shock_value)
            scenarios.append(('Рост_строительства_15%', delta))
    
    # Сценарий 3: IT
    it_idx = find_industry(['IT', 'computer', 'software', 'information', 'J58', 'J61'])
    if it_idx is not None:
        delta = np.zeros(n)
        shock_value = min(Y_base[it_idx] * 0.2, X[it_idx] * 0.2)
        if shock_value > 0:
            delta[it_idx] = abs(shock_value)
            scenarios.append(('Рост_IT_20%', delta))
    
    return scenarios


def get_cached_countries_years() -> dict:
    """Возвращает словарь {страна: [годы]} для которых уже есть кэш"""
    cached = {}
    output_dir = Path("outputs")
    
    if output_dir.exists():
        for folder in output_dir.glob("*_*"):
            if folder.is_dir():
                parts = folder.name.split("_")
                if len(parts) >= 2:
                    country = parts[0]
                    try:
                        year = int(parts[1])
                        if country not in cached:
                            cached[country] = []
                        cached[country].append(year)
                    except ValueError:
                        continue
    
    # Сортируем года
    for country in cached:
        cached[country].sort(reverse=True)
    
    return cached