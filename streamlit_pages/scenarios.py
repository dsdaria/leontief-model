"""
Страница анализа сценариев
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from streamlit_components.charts import create_scenario_analysis
from streamlit_components.layouts import section_title


def render_scenarios(data):
    """Рендеринг страницы сценариев"""
    
    section_title("Анализ экономических сценариев", "🎯")
    
    st.markdown("""
    <div class="info-box">
        🎯 Что показывает анализ сценариев?<br>
        Моделирует, как изменение спроса в одной или нескольких отраслях 
        распространяется по всей экономике через цепочки поставок.
    </div>
    """, unsafe_allow_html=True)
    
    # Генерируем сценарии автоматически, если их нет
    if not data.get('scenarios') or len(data['scenarios']) == 0:
        if data.get('A') is not None and data.get('L') is not None:
            with st.spinner("🔄 Генерация сценариев анализа..."):
                scenarios = generate_scenarios_from_model(data)
                data['scenarios'] = scenarios
    
    if data.get('scenarios') and len(data['scenarios']) > 0:
        scenario_names = list(data['scenarios'].keys())
        if scenario_names:
            selected_scenario = st.selectbox(
                "🎯 Выберите сценарий для анализа:",
                scenario_names
            )
            
            if selected_scenario:
                df = data['scenarios'][selected_scenario]
                
                # Определяем колонки
                value_col = 'Изменение_выпуска' if 'Изменение_выпуска' in df.columns else df.columns[-1]
                industry_col = 'Отрасль' if 'Отрасль' in df.columns else df.columns[0]
                
                total_effect = df[value_col].sum()
                positive_effect = df[df[value_col] > 0][value_col].sum()
                negative_effect = df[df[value_col] < 0][value_col].sum()
                n_positive = (df[value_col] > 0).sum()
                n_negative = (df[value_col] < 0).sum()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Общий эффект", f"{total_effect:,.1f} млн €")
                with col2:
                    st.metric("Рост", f"{positive_effect:,.1f} млн €", delta=f"{n_positive} отраслей")
                with col3:
                    st.metric("Спад", f"{negative_effect:,.1f} млн €", delta=f"{n_negative} отраслей")
                with col4:
                    diff = positive_effect - negative_effect if positive_effect - negative_effect != 0 else 1
                    multiplier = total_effect / diff if diff != 0 else 0
                    st.metric("Мультипликатор", f"{abs(multiplier):.2f}")
                
                st.markdown("---")
                st.markdown(f"### 📊 Детальный анализ: {selected_scenario}")
                
                fig = create_scenario_analysis(df, selected_scenario, top_n=20)
                st.plotly_chart(fig, use_container_width=True)
                
                # Полная таблица
                with st.expander("📋 Полные результаты сценария"):
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Скачать результаты сценария",
                        data=csv,
                        file_name=f"scenario_{selected_scenario}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("ℹ️ Сценарии будут сгенерированы автоматически после загрузки модели")


def generate_scenarios_from_model(data):
    """Генерация сценариев на основе модели"""
    scenarios = {}
    
    industries = data['industries']
    n = len(industries)
    L_matrix = data['L'].values if data['L'] is not None else None
    
    if L_matrix is None:
        return scenarios
    
    # Базовый вектор конечного спроса
    Y_base = np.ones(n) * 100  # 100 млн € базовый спрос на каждую отрасль
    
    # Сценарий 1: Рост промышленности
    industry_indices = find_industry_indices(industries, ['manufacturing', 'metal', 'chemical', 'machinery'])
    if industry_indices:
        delta = np.zeros(n)
        for idx in industry_indices[:3]:
            delta[idx] = 500  # +500 млн €
        delta_X = L_matrix @ delta
        scenarios['Рост промышленности'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
    
    # Сценарий 2: Рост IT-сектора
    it_indices = find_industry_indices(industries, ['IT', 'computer', 'software', 'telecom', 'information'])
    if it_indices:
        delta = np.zeros(n)
        for idx in it_indices[:2]:
            delta[idx] = 300
        delta_X = L_matrix @ delta
        scenarios['Рост IT-сектора'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
    
    # Сценарий 3: Кризис строительства
    const_indices = find_industry_indices(industries, ['construction', 'building'])
    if const_indices:
        delta = np.zeros(n)
        for idx in const_indices[:1]:
            delta[idx] = -400
        delta_X = L_matrix @ delta
        scenarios['Спад строительства'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
    
    # Сценарий 4: Зелёные инвестиции
    energy_indices = find_industry_indices(industries, ['electricity', 'energy', 'renewable', 'solar', 'wind'])
    transport_indices = find_industry_indices(industries, ['transport', 'vehicle', 'car'])
    if energy_indices or transport_indices:
        delta = np.zeros(n)
        for idx in energy_indices[:2]:
            delta[idx] = 200
        for idx in transport_indices[:1]:
            delta[idx] = 300
        delta_X = L_matrix @ delta
        scenarios['Зелёные инвестиции'] = pd.DataFrame({
            'Отрасль': industries,
            'Изменение_выпуска': delta_X
        })
    
    return scenarios


def find_industry_indices(industries, keywords):
    """Поиск индексов отраслей по ключевым словам"""
    indices = []
    for i, ind in enumerate(industries):
        ind_lower = ind.lower()
        for kw in keywords:
            if kw.lower() in ind_lower:
                indices.append(i)
                break
    return indices