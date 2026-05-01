import streamlit as st
import pandas as pd
import psutil
import platform
import time

from streamlit_app.system_metrics import get_system_metrics, render_cpu_gauge, render_memory_chart


def render_system_page(data):
    """Рендеринг страницы системного мониторинга"""
    
    st.markdown("## 💻 Системный мониторинг и производительность")
    
    # Системная информация
    system = get_system_metrics()
    
    st.markdown("### 🖥️ Аппаратное обеспечение")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Операционная система", f"{system['os']} {system['os_release']}")
        st.metric("Процессор", system['processor'][:40] if system['processor'] else "Unknown")
    with col2:
        st.metric("Логических ядер", system['cpu_count_logical'])
        st.metric("Физических ядер", system['cpu_count_physical'])
    with col3:
        st.metric("Оперативная память", f"{system['memory_total']:.1f} GB")
        st.metric("Python версия", system['python_version'])
    
    st.markdown("---")
    st.markdown("### 📊 Текущая загрузка")
    
    col1, col2 = st.columns(2)
    with col1:
        render_cpu_gauge()
    with col2:
        render_memory_chart()
    
    st.markdown("---")
    st.markdown("### ⚡ Производительность матричных расчётов")
    
    if data.get('metadata', {}).get('data_loaded'):
        meta = data['metadata']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Размер матрицы", f"{meta.get('n_industries', 0)}×{meta.get('n_industries', 0)}")
        with col2:
            st.metric("Время расчёта", f"{meta.get('computation_time', 0):.3f} сек")
        with col3:
            st.metric("Использовано потоков", meta.get('n_threads', 1))
        with col4:
            method = meta.get('method_used', 'unknown')
            method_display = "Прямой" if method == "direct_inv" else "Итерационный" if "iterative" in method else method
            st.metric("Метод решения", method_display)
        
        st.markdown("### 📈 Детали производительности")
        
        perf_data = pd.DataFrame([
            {"Параметр": "Память (расчёт)", "Значение": f"{meta.get('memory_usage_mb', 0):.1f} MB"},
            {"Параметр": "Число обусловленности", "Значение": f"{meta.get('condition_number', 0):.2e}" if meta.get('condition_number') else "N/A"},
            {"Параметр": "Количество связей", "Значение": f"{meta.get('n_connections', 0):,}"},
            {"Параметр": "Средний мультипликатор", "Значение": f"{meta.get('avg_output_multiplier', 0):.3f}"}
        ])
        st.dataframe(perf_data, use_container_width=True, hide_index=True)
        
        # Информация о параллельных вычислениях
        st.markdown("### 🚀 Параллельные вычисления")
        
        st.markdown("""
        <div class="info-box">
            <strong>⚡ Оптимизация производительности:</strong><br>
            • Используется библиотека Intel MKL для быстрых матричных операций<br>
            • Многопоточные вычисления задействуют все доступные ядра CPU<br>
            • Для больших матриц (>50) автоматически применяются итерационные методы
        </div>
        """, unsafe_allow_html=True)
        
        # Оценка производительности
        cond = meta.get('condition_number', 0)
        if cond:
            if cond < 1e4:
                st.success("✅ Отличная обусловленность матрицы — результаты надёжны")
            elif cond < 1e8:
                st.warning("⚠️ Удовлетворительная обусловленность — результаты приемлемы")
            else:
                st.error("❌ Плохая обусловленность — результаты могут быть нестабильны")
    else:
        st.info("ℹ️ Данные модели не загружены. Выберите страну и год в боковой панели.")