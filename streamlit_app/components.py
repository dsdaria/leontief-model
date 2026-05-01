"""
Переиспользуемые компоненты Streamlit с поддержкой удаленного решателя
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict

# Добавляем корневую директорию в путь для импорта корневого config.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.config import COLOR_SCHEME
from streamlit_app.remote_client import load_from_remote_solver, USE_REMOTE_SOLVER, REMOTE_SOLVER_URL
from config import AVAILABLE_COUNTRIES, AVAILABLE_YEARS, DEFAULT_COUNTRY, DEFAULT_YEAR
from leontief_model import LeontiefModel
from unified_loader import load_data_with_source, get_source_info


def render_header():
    source_info = get_source_info(st.session_state.get('data_source', 'eurostat'))
    country_name = AVAILABLE_COUNTRIES.get(
        st.session_state.get('selected_country', DEFAULT_COUNTRY), 
        st.session_state.get('selected_country', DEFAULT_COUNTRY)
    )
    
    if USE_REMOTE_SOLVER:
        solver_status = "🌐 Удалённый решатель"
    else:
        solver_status = "💻 Локальный решатель"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #4a90e2 0%, #357abd 50%, #2c5f8a 100%);
                color: white; padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; color: white;">🏭 Модель межотраслевого баланса</h1>
                <p style="margin: 0.5rem 0; color: #e0e0e0;">Метод "Затраты-Выпуск" Василия Леонтьева</p>
                <p style="margin: 0; opacity: 0.9; color: #d0d0d0;">
                    📍 {country_name} ({st.session_state.get('selected_country', DEFAULT_COUNTRY)}) | 
                    📅 {st.session_state.get('selected_year', DEFAULT_YEAR)} | 
                    📊 {source_info['name']} | 
                    {solver_status} | 
                    🔧 {st.session_state.get('threads', 8)} потоков
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 10px;">
                <span style="font-size: 2rem;">{source_info['icon']}</span>
                <p style="margin: 0; color: white;">{source_info['industries']} отраслей</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    from config import EXIOBASE_COUNTRIES, EXIOBASE_YEARS
    
    if USE_REMOTE_SOLVER:
        with st.sidebar:
            try:
                import requests
                response = requests.get(f"{REMOTE_SOLVER_URL}/api/health", timeout=2)
                if response.status_code == 200:
                    st.success("🟢 Удалённый решатель: онлайн")
                else:
                    st.error("🔴 Удалённый решатель: офлайн")
            except:
                st.error("🔴 Удалённый решатель: недоступен")
    
    st.sidebar.markdown("### 📊 Навигация")
    
    pages = ["🏠 Дашборд", "🗺️ Тепловые карты", "📈 Мультипликаторы", "🎯 Сценарии", "🔗 Сетевой анализ", "⚡ Производительность", "⚙️ Система", "ℹ️ О модели"]
    
    for page in pages:
        if st.sidebar.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.current_page = page
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 Параметры")
    
    source_options = ['eurostat', 'exiobase']
    source_labels = {'eurostat': '🇪🇺 Eurostat (64)', 'exiobase': '🌍 EXIOBASE (200+)'}
    
    selected_source = st.sidebar.selectbox(
        "Источник", 
        source_options, 
        format_func=lambda x: source_labels[x],
        key="source_select"
    )
    if selected_source != st.session_state.get('data_source'):
        st.session_state.data_source = selected_source
        st.cache_data.clear()
        st.rerun()
    
    countries = EXIOBASE_COUNTRIES if st.session_state.get('data_source') == 'exiobase' else AVAILABLE_COUNTRIES
    selected_country = st.sidebar.selectbox(
        "Страна", 
        list(countries.keys()), 
        format_func=lambda x: f"{x} - {countries[x]}",
        key="country_select"
    )
    
    years = EXIOBASE_YEARS if st.session_state.get('data_source') == 'exiobase' else AVAILABLE_YEARS
    selected_year = st.sidebar.selectbox(
        "Год", 
        years,
        key="year_select"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Применить", use_container_width=True):
            st.session_state.selected_country = selected_country
            st.session_state.selected_year = selected_year
            st.session_state.data_source = selected_source
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("Сброс", use_container_width=True):
            st.session_state.selected_country = DEFAULT_COUNTRY
            st.session_state.selected_year = DEFAULT_YEAR
            st.session_state.data_source = 'eurostat'
            st.cache_data.clear()
            st.rerun()


def render_footer(data: Dict):
    meta = data.get('metadata', {})
    st.markdown("---")
    
    if USE_REMOTE_SOLVER:
        st.markdown(f"""
        <div style="text-align: center; color: #6c757d; padding: 1rem;">
            <p>🏭 Модель Леонтьева | Данные: {meta.get('source_name', 'Eurostat')} | 
            Потоков: {meta.get('n_threads', st.session_state.get('threads', 8))} | 
            Время расчёта: {meta.get('computation_time', 0):.2f} сек</p>
            <p>🌐 Расчёт выполнен на удалённом сервере | API: {REMOTE_SOLVER_URL}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; color: #6c757d; padding: 1rem;">
            <p>🏭 Модель Леонтьева | Данные: {meta.get('source_name', 'Eurostat')} | 
            Потоков: {meta.get('n_threads', st.session_state.get('threads', 8))} | 
            Время: {meta.get('computation_time', 0):.2f} сек</p>
            <p>© 2025 | Операционные системы | МГТУ им. Н.Э. Баумана</p>
        </div>
        """, unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner="🔄 Загрузка и расчёт модели...")
def load_cached_model_data(country: str, year: int, source: str, 
                           threads: int = 8, method_name: str = "BICGSTAB (итерационный)",
                           tolerance: float = 1e-8, maxiter: int = 1000) -> Dict:
    """Загрузка данных с учётом настроек производительности"""
    
    # Преобразуем метод в формат для сервера
    method_map = {
        "BICGSTAB (итерационный)": "bicgstab",
        "GMRES (итерационный)": "gmres",
        "Прямой (inv)": "direct"
    }
    method_type = method_map.get(method_name, "bicgstab")
    use_iterative = method_type in ["bicgstab", "gmres"]
    
    if USE_REMOTE_SOLVER:
        data = load_from_remote_solver(
            country, year, source, 
            threads, use_iterative, method_type, tolerance, maxiter
        )
        if data:
            return data
        st.warning("⚠️ Удалённый решатель недоступен. Переключение на локальный режим...")
    
    # Локальный расчёт
    start_time = time.time()
    raw_data = load_data_with_source(country, year, source)
    
    model = LeontiefModel(raw_data['Z'], raw_data['X'], raw_data['industries'])
    
    if hasattr(model, 'thread_manager'):
        model.thread_manager.set_num_threads(threads, verbose=False)
    
    if hasattr(model, 'iterative_solver'):
        model.iterative_solver.tol = tolerance
        model.iterative_solver.maxiter = maxiter
    
    A = model.calculate_matrix_A()
    L, metrics = model.calculate_leontief_matrix_parallel(
        use_iterative=use_iterative,
        n_threads=threads,
        method=method_type
    )
    multipliers_df = model.calculate_multipliers()
    
    scenarios = generate_scenarios_local(model, raw_data['industries'], raw_data['Y'], L)
    
    n_industries = raw_data['n']
    n_connections = int((A > 0).sum().sum())
    avg_multiplier = float(multipliers_df['Мультипликатор_выпуска'].mean())
    max_multiplier = float(multipliers_df['Мультипликатор_выпуска'].max())
    
    return {
        'A': pd.DataFrame(A, index=raw_data['industries'], columns=raw_data['industries']),
        'L': pd.DataFrame(L, index=raw_data['industries'], columns=raw_data['industries']),
        'multipliers': multipliers_df,
        'scenarios': scenarios,
        'industries': raw_data['industries'],
        'metadata': {
            'data_loaded': True,
            'n_industries': n_industries,
            'n_connections': n_connections,
            'avg_output_multiplier': avg_multiplier,
            'max_output_multiplier': max_multiplier,
            'source': source,
            'source_name': get_source_info(source)['name'],
            'country_code': country,
            'year': year,
            'computation_time': time.time() - start_time,
            'condition_number': metrics.condition_number if metrics else None,
            'method_used': metrics.method_used if metrics else ('iterative' if use_iterative else 'direct'),
            'n_threads': metrics.n_threads if metrics else threads,
            'memory_usage_mb': metrics.memory_usage_mb if metrics else 0
        }
    }


def generate_scenarios_local(model, industries, Y_base, L_matrix):
    """Локальная генерация сценариев"""
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
    
    man_indices = find_indices(['manufacturing', 'metal', 'chemical', 'machinery'])
    if man_indices:
        delta = np.zeros(n)
        for idx in man_indices[:3]:
            delta[idx] = 500
        delta_X = L @ delta
        scenarios['Рост промышленности'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
    
    it_indices = find_indices(['it', 'computer', 'software', 'telecom'])
    if it_indices:
        delta = np.zeros(n)
        for idx in it_indices[:2]:
            delta[idx] = 300
        delta_X = L @ delta
        scenarios['Рост IT-сектора'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
    
    const_indices = find_indices(['construction', 'building'])
    if const_indices:
        delta = np.zeros(n)
        for idx in const_indices[:1]:
            delta[idx] = -400
        delta_X = L @ delta
        scenarios['Спад строительства'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
    
    return scenarios