"""
Удаленный решатель модели Леонтьева
REST API сервер для вычислений
Запуск: python remote_solver.py
Деплой: Render.com (gunicorn remote_solver:app)
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import time
import json
import threading
from datetime import datetime
import logging

from leontief_model import LeontiefModel
from data_loader import EurostatDataLoader
from exiobase_loader import EXIOBASELoader
from config import AVAILABLE_COUNTRIES, AVAILABLE_YEARS, EXIOBASE_COUNTRIES, EXIOBASE_YEARS

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Разрешаем кросс-доменные запросы (важно для Streamlit Cloud)

# Кэш для результатов (in-memory)
cache = {}
cache_lock = threading.Lock()

# Статистика запросов
request_stats = {
    'total_requests': 0,
    'cache_hits': 0,
    'avg_response_time': 0,
    'start_time': datetime.now().isoformat()
}


@app.route('/api/health', methods=['GET'])
def health():
    """Проверка состояния сервера"""
    return jsonify({
        'status': 'ok',
        'message': 'Remote Leontief Model Solver is running',
        'timestamp': datetime.now().isoformat(),
        'stats': request_stats
    })


@app.route('/api/compute', methods=['POST'])
def compute():
    """Расчёт модели на удаленном сервере с поддержкой настроек производительности"""
    start_time = time.time()
    request_stats['total_requests'] += 1
    
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Основные параметры
        country = data.get('country', 'FR')
        year = data.get('year', 2020)
        source = data.get('source', 'eurostat')
        
        # ========== НАСТРОЙКИ ПРОИЗВОДИТЕЛЬНОСТИ ==========
        threads = data.get('threads', 8)
        use_iterative = data.get('use_iterative', True)
        method = data.get('method', 'bicgstab')
        tolerance = data.get('tolerance', 1e-8)
        maxiter = data.get('maxiter', 1000)
        
        logger.info(f"📊 Запрос на расчёт: {country} {year} {source}")
        logger.info(f"⚙️ Настройки: threads={threads}, method={method}, use_iterative={use_iterative}, tolerance={tolerance}, maxiter={maxiter}")
        
        # Ключ кэша с учётом настроек
        cache_key = f"{country}_{year}_{source}_{threads}_{method}_{tolerance}_{maxiter}"
        
        # Проверяем кэш
        with cache_lock:
            if cache_key in cache:
                request_stats['cache_hits'] += 1
                logger.info(f"✅ Возвращено из кэша: {cache_key}")
                return jsonify(cache[cache_key])
        
        # Загрузка данных
        logger.info(f"📥 Загрузка данных для {country} {year} из {source}...")
        
        if source == 'exiobase':
            loader = EXIOBASELoader(country, year)
        else:
            loader = EurostatDataLoader(country, year)
        
        raw_data = loader.get_input_output_tables()
        
        if not raw_data or raw_data.get('n', 0) == 0:
            return jsonify({'error': 'No data available for this country/year'}), 404
        
        n_industries = raw_data['n']
        logger.info(f"📊 Загружено {n_industries} отраслей")
        
        # Расчёт модели с переданными настройками
        logger.info(f"🔢 Расчёт модели для {n_industries} отраслей с {threads} потоками...")
        
        model = LeontiefModel(raw_data['Z'], raw_data['X'], raw_data['industries'])
        
        # Применяем настройки производительности
        model.thread_manager.set_num_threads(threads, verbose=False)
        
        # Настройка итерационного решателя
        if hasattr(model, 'iterative_solver'):
            model.iterative_solver.tol = tolerance
            model.iterative_solver.maxiter = maxiter
            logger.info(f"🔧 Итерационный решатель: tol={tolerance}, maxiter={maxiter}")
        
        # Матрица A
        logger.info("📐 Расчёт матрицы A...")
        A = model.calculate_matrix_A()
        
        # Матрица L (с переданными настройками)
        logger.info(f"🔢 Расчёт матрицы L (метод: {method if use_iterative else 'direct'})...")
        L, metrics = model.calculate_leontief_matrix_parallel(
            use_iterative=use_iterative,
            n_threads=threads,
            method=method
        )
        
        # Мультипликаторы
        logger.info("📈 Расчёт мультипликаторов...")
        multipliers_df = model.calculate_multipliers()
        
        # Генерация сценариев
        logger.info("🎯 Генерация сценариев...")
        scenarios = generate_scenarios(model, raw_data['industries'], raw_data['Y'], L)
        
        # Подготовка результата для сериализации
        result = {
            'success': True,
            'data': {
                'A': A.tolist(),
                'L': L.tolist(),
                'multipliers': multipliers_df.to_dict('records'),
                'scenarios': {name: df.to_dict('records') for name, df in scenarios.items()},
                'industries': raw_data['industries'],
                'n_industries': n_industries,
                'metadata': {
                    'condition_number': metrics.condition_number if metrics else None,
                    'method_used': metrics.method_used if metrics else ('iterative' if use_iterative else 'direct'),
                    'n_threads': metrics.n_threads if metrics else threads,
                    'memory_usage_mb': metrics.memory_usage_mb if metrics else 0,
                    'computation_time': metrics.inversion_time if metrics else 0,
                    'total_time': time.time() - start_time,
                    'source': source,
                    'country': country,
                    'year': year,
                    'settings': {
                        'threads': threads,
                        'method': method,
                        'tolerance': tolerance,
                        'maxiter': maxiter
                    }
                }
            }
        }
        
        # Сохраняем в кэш
        with cache_lock:
            cache[cache_key] = result
            logger.info(f"💾 Сохранено в кэш: {cache_key}, размер кэша: {len(cache)}")
        
        # Обновляем статистику
        response_time = time.time() - start_time
        request_stats['avg_response_time'] = (
            (request_stats['avg_response_time'] * (request_stats['total_requests'] - 1) + response_time) 
            / request_stats['total_requests']
        )
        
        logger.info(f"✅ Расчёт завершён за {response_time:.2f} сек")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/available', methods=['GET'])
def get_available():
    """Возвращает доступные страны и годы"""
    return jsonify({
        'eurostat': {
            'countries': AVAILABLE_COUNTRIES,
            'years': AVAILABLE_YEARS
        },
        'exiobase': {
            'countries': EXIOBASE_COUNTRIES,
            'years': EXIOBASE_YEARS
        }
    })


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Очистка кэша"""
    with cache_lock:
        cache_size = len(cache)
        cache.clear()
    logger.info(f"🗑️ Кэш очищен. Удалено {cache_size} записей")
    return jsonify({'status': 'ok', 'message': 'Cache cleared', 'cleared': cache_size})


@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Статистика кэша"""
    with cache_lock:
        return jsonify({
            'cache_size': len(cache),
            'cache_hits': request_stats['cache_hits'],
            'total_requests': request_stats['total_requests'],
            'hit_rate': request_stats['cache_hits'] / request_stats['total_requests'] if request_stats['total_requests'] > 0 else 0
        })


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Возвращает текущие настройки сервера"""
    return jsonify({
        'available_threads': os.cpu_count(),
        'default_threads': 8,
        'available_methods': ['bicgstab', 'gmres', 'direct'],
        'default_method': 'bicgstab',
        'tolerance_range': [1e-12, 1e-10, 1e-8, 1e-6],
        'maxiter_range': [100, 500, 1000, 2000, 5000]
    })


def generate_scenarios(model, industries, Y_base, L_matrix):
    """Генерация сценариев"""
    import numpy as np
    
    scenarios = {}
    n = len(industries)
    
    if L_matrix is None:
        return scenarios
    
    if isinstance(L_matrix, pd.DataFrame):
        L = L_matrix.values
    else:
        L = L_matrix
    
    def find_indices(keywords):
        indices = []
        for i, ind in enumerate(industries):
            ind_lower = ind.lower()
            for kw in keywords:
                if kw.lower() in ind_lower:
                    indices.append(i)
                    break
        return indices
    
    # Сценарий 1: Рост промышленности
    man_indices = find_indices(['manufacturing', 'metal', 'chemical', 'machinery', 'industrial'])
    if man_indices:
        delta = np.zeros(n)
        shock_sum = 0
        for idx in man_indices[:5]:
            val = 300
            delta[idx] = val
            shock_sum += val
        delta_X = L @ delta
        scenarios['Рост промышленности'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
        logger.info(f"   📊 Сценарий 'Рост промышленности': шок {shock_sum:.0f} млн €")
    
    # Сценарий 2: Рост IT
    it_indices = find_indices(['it', 'computer', 'software', 'telecom', 'information', 'technology'])
    if it_indices:
        delta = np.zeros(n)
        shock_sum = 0
        for idx in it_indices[:3]:
            val = 200
            delta[idx] = val
            shock_sum += val
        delta_X = L @ delta
        scenarios['Рост IT-сектора'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
        logger.info(f"   📊 Сценарий 'Рост IT-сектора': шок {shock_sum:.0f} млн €")
    
    # Сценарий 3: Спад строительства
    const_indices = find_indices(['construction', 'building', 'civil', 'architecture'])
    if const_indices:
        delta = np.zeros(n)
        shock_sum = 0
        for idx in const_indices[:2]:
            val = -300
            delta[idx] = val
            shock_sum += val
        delta_X = L @ delta
        scenarios['Спад строительства'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
        logger.info(f"   📊 Сценарий 'Спад строительства': шок {shock_sum:.0f} млн €")
    
    # Сценарий 4: Зелёные инвестиции
    energy_indices = find_indices(['electricity', 'energy', 'renewable', 'solar', 'wind', 'nuclear', 'power'])
    transport_indices = find_indices(['transport', 'vehicle', 'car', 'motor', 'automotive', 'railway'])
    if energy_indices or transport_indices:
        delta = np.zeros(n)
        shock_sum = 0
        for idx in energy_indices[:3]:
            val = 150
            delta[idx] = val
            shock_sum += val
        for idx in transport_indices[:2]:
            val = 200
            delta[idx] = val
            shock_sum += val
        delta_X = L @ delta
        scenarios['Зелёные инвестиции'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
        logger.info(f"   📊 Сценарий 'Зелёные инвестиции': шок {shock_sum:.0f} млн €")
    
    # Сценарий 5: Рост туризма и услуг
    service_indices = find_indices(['hotel', 'restaurant', 'tourism', 'hospitality', 'accommodation', 'travel'])
    if service_indices:
        delta = np.zeros(n)
        shock_sum = 0
        for idx in service_indices[:3]:
            val = 250
            delta[idx] = val
            shock_sum += val
        delta_X = L @ delta
        scenarios['Рост туризма'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
        logger.info(f"   📊 Сценарий 'Рост туризма': шок {shock_sum:.0f} млн €")
    
    return scenarios


if __name__ == '__main__':
    # Для локального запуска: порт 5000
    # Для Render: порт из переменной окружения PORT
    port = int(os.environ.get("PORT", 5000))
    
    print("=" * 60)
    print("🚀 ЗАПУСК УДАЛЁННОГО РЕШАТЕЛЯ МОДЕЛИ ЛЕОНТЬЕВА")
    print("=" * 60)
    print(f"📡 API сервер запущен на http://0.0.0.0:{port}")
    print(f"📋 Доступные эндпоинты:")
    print(f"   GET  /api/health      - проверка состояния")
    print(f"   POST /api/compute     - расчёт модели (с поддержкой настроек)")
    print(f"   GET  /api/available   - доступные данные")
    print(f"   GET  /api/cache/stats - статистика кэша")
    print(f"   POST /api/cache/clear - очистка кэша")
    print(f"   GET  /api/settings    - настройки сервера")
    print("=" * 60)
    print(f"💻 Доступно ядер CPU: {os.cpu_count()}")
    print(f"🔧 По умолчанию: 8 потоков, метод BICGSTAB, точность 1e-8")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)