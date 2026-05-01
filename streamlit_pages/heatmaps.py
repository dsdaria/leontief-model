"""
Страница тепловых карт - с динамическим размером
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from streamlit_components.charts import create_heatmap_plotly
from streamlit_components.layouts import section_title


def render_heatmaps(data):
    """Рендеринг страницы тепловых карт"""
    
    section_title("Визуализация матриц затрат", "🗺️")
    
    # Определяем максимальное количество отраслей в зависимости от источника
    n_industries = data['metadata'].get('n_industries', 0)
    source = data['metadata'].get('source', 'eurostat')
    
    if source == 'exiobase':
        max_industries = min(200, n_industries)
        default_show = min(50, n_industries)
    else:
        max_industries = min(64, n_industries)
        default_show = min(30, n_industries)
    
    st.markdown(f"""
    <div class="info-box">
        💡 Как читать тепловые карты:<br>
        • Каждая клетка показывает взаимосвязь между двумя отраслями<br>
        • Чем ярче цвет — тем сильнее экономическая связь<br>
        • Диагональ — самопотребление отрасли<br>
        • Всего отраслей в модели: {n_industries} (источник: {source.upper()})
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs([
        "📐 Матрица прямых затрат (A)",
        "🔗 Матрица полных затрат (L)"
    ])
    
    with tab1:
        if data['A'] is not None:
            col1, col2 = st.columns([3, 1])
            with col2:
                n_show = st.slider(
                    "Количество отраслей:", 
                    min_value=5, 
                    max_value=max_industries, 
                    value=min(default_show, max_industries),
                    key="a_slider"
                )
                use_log = st.checkbox("Логарифмическая шкала", value=True, key="a_log")
                st.caption(f"📊 Показано {n_show} из {n_industries} отраслей")
            
            with col1:
                fig = create_heatmap_plotly(
                    data['A'],
                    f"Матрица прямых затрат A ({n_show}/{n_industries} отраслей)",
                    colorscale='RdBu',
                    n_show=n_show,
                    use_log=use_log
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Статистика
            nonzero = (data['A'].values > 0).sum()
            sparsity = 100 * (1 - nonzero / (n_industries ** 2))
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <strong>📊 Статистика матрицы:</strong><br>
                • Разреженность: {sparsity:.1f}% нулевых элементов<br>
                • Ненулевых связей: {nonzero:,}<br>
                • Максимальное значение: {data['A'].values.max():.4f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Матрица A не загружена")
    
    with tab2:
        if data['L'] is not None:
            col1, col2 = st.columns([3, 1])
            with col2:
                n_show = st.slider(
                    "Количество отраслей:", 
                    min_value=5, 
                    max_value=max_industries, 
                    value=min(default_show, max_industries),
                    key="l_slider"
                )
                use_log = st.checkbox("Логарифмическая шкала", value=True, key="l_log")
                st.caption(f"📊 Показано {n_show} из {n_industries} отраслей")
            
            with col1:
                fig = create_heatmap_plotly(
                    data['L'],
                    f"Матрица полных затрат L ({n_show}/{n_industries} отраслей)",
                    colorscale='Viridis',
                    n_show=n_show,
                    use_log=use_log
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if data['metadata'].get('condition_number'):
                cond = data['metadata']['condition_number']
                if cond < 1e4:
                    cond_status = "✅ Отличная обусловленность"
                    cond_color = "#28a745"
                elif cond < 1e8:
                    cond_status = "⚠️ Удовлетворительная обусловленность"
                    cond_color = "#ffc107"
                else:
                    cond_status = "❌ Плохая обусловленность (требуется внимание)"
                    cond_color = "#dc3545"
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid {cond_color};">
                    <strong>🔢 Число обусловленности:</strong> {cond:.2e}<br>
                    <strong>Статус:</strong> {cond_status}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Матрица L не загружена")