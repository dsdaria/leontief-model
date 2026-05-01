"""
Страница анализа мультипликаторов
"""

import streamlit as st
import plotly.express as px

from streamlit_components.charts import create_bar_chart_horizontal
from streamlit_components.layouts import section_title


def render_multipliers(data):
    """Рендеринг страницы мультипликаторов"""
    
    section_title("Анализ мультипликаторов", "📈")
    
    if data['multipliers'] is not None:
        # Исправленный info_box без лишних тегов strong
        st.markdown("""
        <div class="info-box">
            🎯 Что такое мультипликатор?<br>
            Мультипликатор показывает эффект распространения изменений в экономике.<br>
            Мультипликатор = 2.5 означает, что увеличение спроса на €1 млн 
            создаст дополнительный выпуск на €2.5 млн по всей экономике.
        </div>
        """, unsafe_allow_html=True)
        
        # Поиск отрасли
        search_term = st.text_input("🔍 Поиск отрасли:", "")
        
        if search_term:
            filtered = data['multipliers'][
                data['multipliers']['Отрасль'].str.contains(search_term, case=False)
            ]
            st.dataframe(filtered, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 Мультипликаторы выпуска")
            st.markdown("*Отрасли, создающие наибольший эффект в экономике*")
            
            fig = create_bar_chart_horizontal(
                data['multipliers'],
                'Мультипликатор_выпуска',
                'Отрасль',
                'Топ-15: Эффект на выпуск',
                top_n=15,
                color_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📥 Мультипликаторы затрат")
            st.markdown("*Отрасли, наиболее зависимые от поставщиков*")
            
            fig = create_bar_chart_horizontal(
                data['multipliers'],
                'Мультипликатор_затрат',
                'Отрасль',
                'Топ-15: Зависимость от поставок',
                top_n=15,
                color_scale='Oranges'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.markdown("### 📊 Соотношение мультипликаторов")
        fig_scatter = px.scatter(
            data['multipliers'],
            x='Мультипликатор_затрат',
            y='Мультипликатор_выпуска',
            hover_data=['Отрасль'],
            title='Выпуск vs Затраты',
            color_discrete_sequence=['#2a5298']
        )
        fig_scatter.add_hline(y=1, line_dash="dash", line_color="red")
        fig_scatter.add_vline(x=1, line_dash="dash", line_color="red")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Полная таблица
        with st.expander("📋 Полная таблица мультипликаторов"):
            st.dataframe(
                data['multipliers'].style
                    .background_gradient(subset=['Мультипликатор_выпуска'], cmap='Greens')
                    .background_gradient(subset=['Мультипликатор_затрат'], cmap='Oranges')
                    .format({
                        'Мультипликатор_выпуска': '{:.3f}',
                        'Мультипликатор_затрат': '{:.3f}'
                    }),
                use_container_width=True
            )
            
            csv = data['multipliers'].to_csv(index=False)
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name="multipliers.csv",
                mime="text/csv"
            )
    else:
        st.warning("Данные о мультипликаторах не загружены")