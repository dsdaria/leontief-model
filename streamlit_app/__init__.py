"""
Главное приложение модели Леонтьева
Профессиональный интерфейс с мониторингом ОС
"""

import streamlit as st
import warnings
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

# Импорт модулей Streamlit
from streamlit_app.config import PAGE_CONFIG
from streamlit_app.styles import inject_custom_css
from streamlit_app.components import (
    render_header,
    render_sidebar,
    render_footer,
    load_cached_model_data
)
from streamlit_app.system_metrics import (
    render_system_panel,
    render_performance_dashboard
)

# Импорт страниц
from streamlit_pages.dashboard import render_dashboard
from streamlit_pages.heatmaps import render_heatmaps
from streamlit_pages.multipliers import render_multipliers
from streamlit_pages.scenarios import render_scenarios
from streamlit_pages.network import render_network_analysis
from streamlit_pages.about import render_about
from streamlit_pages.performance import render_performance

from streamlit_app.pages.system import render_system_page

warnings.filterwarnings('ignore')


# ==================== КОНФИГУРАЦИЯ ====================
st.set_page_config(**PAGE_CONFIG)
inject_custom_css()


# ==================== ИНИЦИАЛИЗАЦИЯ SESSION STATE ====================
if 'selected_country' not in st.session_state:
    from config import DEFAULT_COUNTRY
    st.session_state.selected_country = DEFAULT_COUNTRY
if 'selected_year' not in st.session_state:
    from config import DEFAULT_YEAR
    st.session_state.selected_year = DEFAULT_YEAR
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'eurostat'
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Дашборд"
if 'last_source' not in st.session_state:
    st.session_state.last_source = st.session_state.data_source
if 'last_country' not in st.session_state:
    st.session_state.last_country = st.session_state.selected_country
if 'last_year' not in st.session_state:
    st.session_state.last_year = st.session_state.selected_year

# ========== НАСТРОЙКИ ПРОИЗВОДИТЕЛЬНОСТИ ==========
if 'threads' not in st.session_state:
    st.session_state.threads = 8
if 'method' not in st.session_state:
    st.session_state.method = "BICGSTAB (итерационный)"
if 'tolerance' not in st.session_state:
    st.session_state.tolerance = 1e-8
if 'maxiter' not in st.session_state:
    st.session_state.maxiter = 1000


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    """Главная функция приложения"""
    
    # Проверяем, изменился ли источник данных
    source_changed = (st.session_state.data_source != st.session_state.last_source or
                      st.session_state.selected_country != st.session_state.last_country or
                      st.session_state.selected_year != st.session_state.last_year)
    
    if source_changed:
        st.session_state.last_source = st.session_state.data_source
        st.session_state.last_country = st.session_state.selected_country
        st.session_state.last_year = st.session_state.selected_year
        st.cache_data.clear()
    
    render_header()
    render_sidebar()
    
    # Передаём настройки производительности в загрузку модели
    data = load_cached_model_data(
        st.session_state.selected_country,
        st.session_state.selected_year,
        st.session_state.data_source,
        st.session_state.threads,
        st.session_state.method,
        st.session_state.tolerance,
        st.session_state.maxiter
    )
    st.session_state.data = data
    
    with st.expander("💻 Системные метрики", expanded=False):
        render_system_panel()
        if data.get('metadata', {}).get('data_loaded'):
            render_performance_dashboard(data)
    
    st.markdown("---")
    
    pages = {
        "🏠 Дашборд": render_dashboard,
        "🗺️ Тепловые карты": render_heatmaps,
        "📈 Мультипликаторы": render_multipliers,
        "🎯 Сценарии": render_scenarios,
        "🔗 Сетевой анализ": render_network_analysis,
        "⚡ Производительность": render_performance,
        "⚙️ Система": render_system_page,
        "ℹ️ О модели": render_about
    }
    
    if st.session_state.current_page in pages:
        pages[st.session_state.current_page](data)
    
    render_footer(data)


if __name__ == "__main__":
    main()