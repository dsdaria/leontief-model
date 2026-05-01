"""
Мониторинг системных характеристик
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psutil
import platform
from typing import Dict


def get_system_metrics() -> Dict:
    return {
        'os': platform.system(),
        'os_release': platform.release(),
        'processor': platform.processor(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'cpu_percent': psutil.cpu_percent(interval=0.5),
        'memory': psutil.virtual_memory(),
        'memory_total': psutil.virtual_memory().total / (1024**3),
        'memory_used': psutil.virtual_memory().used / (1024**3),
        'memory_percent': psutil.virtual_memory().percent,
        'python_version': platform.python_version(),
        'hostname': platform.node()
    }


def render_system_panel():
    """Отображение системной панели"""
    system = get_system_metrics()
    
    st.markdown("### 💻 Системные характеристики")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ОС", f"{system['os']} {system['os_release']}")
    with col2:
        st.metric("CPU", f"{system['cpu_percent']:.0f}%", f"{system['cpu_count_logical']} ядер")
    with col3:
        st.metric("RAM", f"{system['memory_percent']:.0f}%", f"{system['memory_used']:.1f}/{system['memory_total']:.1f} GB")
    with col4:
        st.metric("Python", system['python_version'])


def render_cpu_gauge():
    """Индикатор CPU"""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=cpu_percent,
        title={"text": "CPU", "font": {"size": 20}},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": "#4a90e2"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "lightgray",
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 80], "color": "yellow"},
                {"range": [80, 100], "color": "salmon"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90
            }
        },
        number={"font": {"size": 40, "color": "#2c3e50"}}
    ))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)


def render_performance_dashboard(data: Dict):
    """Панель производительности"""
    meta = data.get('metadata', {})
    st.markdown("### ⚡ Производительность расчётов")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Размер матрицы", f"{meta.get('n_industries', 0)}×{meta.get('n_industries', 0)}")
    with col2:
        st.metric("Время расчёта", f"{meta.get('computation_time', 0):.2f} сек")
    with col3:
        st.metric("Использовано потоков", meta.get('n_threads', 1))
    with col4:
        method = meta.get('method_used', 'unknown')
        method_display = "Прямой" if method == "direct_inv" else "Итерационный" if "iterative" in method else method[:15]
        st.metric("Метод решения", method_display)
    
    if meta.get('condition_number'):
        cond = meta['condition_number']
        if cond < 1e4:
            st.success(f"✅ Число обусловленности: {cond:.2e} — отличное")
        elif cond < 1e8:
            st.warning(f"⚠️ Число обусловленности: {cond:.2e} — удовлетворительное")
        else:
            st.error(f"❌ Число обусловленности: {cond:.2e} — требуется внимание")