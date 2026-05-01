"""
Страница сетевого анализа - с динамическим размером
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from streamlit_components.layouts import section_title


def render_network_analysis(data):
    """Рендеринг страницы сетевого анализа"""
    
    section_title("Сетевой анализ взаимосвязей", "🔗")
    
    if data['A'] is None:
        st.warning("Матрица A не загружена. Пожалуйста, выберите страну и год.")
        return
    
    n_industries = data['metadata'].get('n_industries', 0)
    source = data['metadata'].get('source', 'eurostat')
    
    # Определяем максимальное количество для отображения
    if source == 'exiobase':
        max_sectors = min(50, n_industries)
    else:
        max_sectors = min(30, n_industries)
    
    st.markdown(f"""
    <div class="info-box">
        🔗 Анализ показывает, какие отрасли являются ключевыми поставщиками и потребителями.<br>
        Чем выше значение, тем сильнее экономическая взаимосвязь между отраслями.<br>
        Всего отраслей в модели: {n_industries} (источник: {source.upper()})
    </div>
    """, unsafe_allow_html=True)
    
    # Настройки
    col1, col2, col3 = st.columns(3)
    with col1:
        n_sectors = st.selectbox(
            "Количество отраслей:", 
            options=[10, 15, 20, 25, 30, 40, 50],
            index=1,
            help=f"Максимум: {max_sectors} отраслей"
        )
        n_sectors = min(n_sectors, max_sectors)
    with col2:
        metric_type = st.selectbox("Показатель:", ["Прямые затраты (A)", "Мультипликаторы"])
    with col3:
        sort_by = st.selectbox("Сортировка:", ["По влиянию", "По зависимости", "По алфавиту"])
    
    # Берём топ отрасли
    if metric_type == "Мультипликаторы" and data['multipliers'] is not None:
        if sort_by == "По влиянию":
            top_industries = data['multipliers'].nlargest(n_sectors, 'Мультипликатор_выпуска')['Отрасль'].values
        elif sort_by == "По зависимости":
            top_industries = data['multipliers'].nlargest(n_sectors, 'Мультипликатор_затрат')['Отрасль'].values
        else:
            top_industries = data['multipliers']['Отрасль'].values[:n_sectors]
    else:
        # Берём отрасли с наибольшим суммарным объёмом
        sums = data['A'].sum().sort_values(ascending=False)
        top_industries = sums.head(n_sectors).index
    
    # Показываем мультипликаторы
    if metric_type == "Мультипликаторы" and data['multipliers'] is not None:
        st.subheader("📊 Мультипликаторы отраслей")
        
        display_df = data['multipliers'].set_index('Отрасль')
        display_df = display_df.loc[top_industries]
        
        fig = go.Figure(data=[
            go.Bar(name='Мультипликатор выпуска', x=display_df.index, y=display_df['Мультипликатор_выпуска'], 
                   marker_color='#4a90e2', text=display_df['Мультипликатор_выпуска'].round(3), textposition='outside'),
            go.Bar(name='Мультипликатор затрат', x=display_df.index, y=display_df['Мультипликатор_затрат'], 
                   marker_color='#ff9800', text=display_df['Мультипликатор_затрат'].round(3), textposition='outside')
        ])
        fig.update_layout(
            title=f"Топ-{n_sectors} отраслей по мультипликаторам",
            barmode='group',
            height=500,
            xaxis_tickangle=-45,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Таблица с данными
        st.subheader("📋 Подробные данные")
        st.dataframe(
            data['multipliers'].loc[data['multipliers']['Отрасль'].isin(top_industries)].style
                .background_gradient(subset=['Мультипликатор_выпуска'], cmap='Blues')
                .background_gradient(subset=['Мультипликатор_затрат'], cmap='Oranges'),
            use_container_width=True
        )
        return
    
    # Матрица для отображения
    matrix_data = data['A'].loc[top_industries, top_industries]
    
    # Тепловая карта связей
    st.subheader(f"🔥 Карта связей (матрица прямых затрат A)")
    
    fig = go.Figure(data=go.Heatmap(
        z=np.log1p(matrix_data.values),
        x=matrix_data.columns,
        y=matrix_data.index,
        colorscale='Blues',
        text=matrix_data.values.round(4),
        texttemplate='%{text}',
        textfont={"size": 9},
        hoverongaps=False,
        colorbar_title="log(1 + значение)"
    ))
    fig.update_layout(
        title=f"Взаимосвязи между отраслями (показано {n_sectors} из {n_industries})",
        height=max(500, n_sectors * 15),
        xaxis_tickangle=-45,
        xaxis_title="Отрасль-потребитель",
        yaxis_title="Отрасль-производитель",
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Анализ ключевых отраслей
    st.subheader("📊 Ключевые отрасли экономики")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Самые важные поставщики
        out_degree = matrix_data.sum(axis=1).sort_values(ascending=False)
        fig_out = go.Figure(data=[
            go.Bar(
                x=out_degree.values,
                y=out_degree.index,
                orientation='h',
                marker_color='#4a90e2',
                text=out_degree.values.round(3),
                textposition='outside'
            )
        ])
        fig_out.update_layout(
            title="🏭 Ключевые поставщики",
            height=400,
            margin=dict(l=200, r=20, t=40, b=20),
            xaxis_title="Суммарный объём поставок",
            yaxis_title="Отрасль"
        )
        st.plotly_chart(fig_out, use_container_width=True)
    
    with col2:
        # Самые важные потребители
        in_degree = matrix_data.sum(axis=0).sort_values(ascending=False)
        fig_in = go.Figure(data=[
            go.Bar(
                x=in_degree.values,
                y=in_degree.index,
                orientation='h',
                marker_color='#ff9800',
                text=in_degree.values.round(3),
                textposition='outside'
            )
        ])
        fig_in.update_layout(
            title="📥 Ключевые потребители",
            height=400,
            margin=dict(l=200, r=20, t=40, b=20),
            xaxis_title="Суммарный объём потребления",
            yaxis_title="Отрасль"
        )
        st.plotly_chart(fig_in, use_container_width=True)
    
    # Таблица связей
    with st.expander("🔍 Детальная таблица связей"):
        st.dataframe(matrix_data.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # Статистика сети
    total_links = (matrix_data.values > 0).sum()
    density = total_links / (len(top_industries) * (len(top_industries) - 1)) if len(top_industries) > 1 else 0
    avg_strength = matrix_data.values[matrix_data.values > 0].mean() if total_links > 0 else 0
    
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
        <strong>🌐 Статистика сети:</strong><br>
        • Количество отраслей: {len(top_industries)} (всего в модели: {n_industries})<br>
        • Количество связей: {total_links}<br>
        • Плотность сети: {density:.2%}<br>
        • Средняя сила связи: {avg_strength:.4f}
    </div>
    """, unsafe_allow_html=True)