"""
Страница дашборда
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from streamlit_components.metrics import render_metrics_row
from streamlit_components.charts import create_bar_chart_horizontal
from streamlit_components.layouts import section_title


def render_dashboard(data):
    """Рендеринг страницы дашборда"""
    
    section_title("Обзор экономической модели", "📊")
    
    # Получаем метрики из metadata
    n_industries = data['metadata'].get('n_industries', 0)
    avg_multiplier = data['metadata'].get('avg_output_multiplier', 0)
    n_connections = data['metadata'].get('n_connections', 0)
    n_scenarios = len(data.get('scenarios', {}))
    
    metrics = [
        {
            'icon': '🏭',
            'value': str(n_industries),
            'label': 'Отраслей',
            'sublabel': 'в модели'
        },
        {
            'icon': '📈',
            'value': f"{avg_multiplier:.2f}",
            'label': 'Средний мультипликатор',
            'sublabel': 'единиц выпуска'
        },
        {
            'icon': '🔗',
            'value': f"{n_connections:,}",
            'label': 'Связей',
            'sublabel': 'между отраслями'
        },
        {
            'icon': '🎯',
            'value': str(n_scenarios),
            'label': 'Сценариев',
            'sublabel': 'для анализа'
        }
    ]
    
    render_metrics_row(metrics)
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🏆 Ключевые отрасли экономики")
        
        if data['multipliers'] is not None:
            tab1, tab2 = st.tabs(["📊 График", "📋 Таблица"])
            
            with tab1:
                fig = create_bar_chart_horizontal(
                    data['multipliers'],
                    'Мультипликатор_выпуска',
                    'Отрасль',
                    'Топ-10 отраслей по влиянию на экономику',
                    top_n=10,
                    color_scale='Greens'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.dataframe(
                    data['multipliers'].nlargest(10, 'Мультипликатор_выпуска')[
                        ['Отрасль', 'Мультипликатор_выпуска', 'Мультипликатор_затрат']
                    ].style.format({
                        'Мультипликатор_выпуска': '{:.3f}',
                        'Мультипликатор_затрат': '{:.3f}'
                    }),
                    use_container_width=True
                )
        else:
            st.warning("Данные о мультипликаторах не загружены")
    
    with col2:
        st.markdown("### 📊 Распределение мультипликаторов")
        
        if data['multipliers'] is not None:
            fig = px.histogram(
                data['multipliers'],
                x='Мультипликатор_выпуска',
                nbins=20,
                title='Гистограмма мультипликаторов',
                color_discrete_sequence=['#667eea']
            )
            fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Базовый уровень")
            fig.update_layout(template='plotly_white', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            mults = data['multipliers']['Мультипликатор_выпуска']
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                        padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <strong>Статистика:</strong><br>
                • Минимум: {mults.min():.3f}<br>
                • Медиана: {mults.median():.3f}<br>
                • Максимум: {mults.max():.3f}<br>
                • Стандартное отклонение: {mults.std():.3f}
            </div>
            """, unsafe_allow_html=True)