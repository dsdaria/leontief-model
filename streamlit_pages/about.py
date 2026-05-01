"""
Страница информации о модели - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

import streamlit as st


def render_about(data):
    """Рендеринг страницы информации о модели"""
    
    st.markdown("""
    <style>
    .about-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .about-title {
        color: #4a90e2;
        border-left: 4px solid #4a90e2;
        padding-left: 1rem;
        margin-bottom: 1.5rem;
    }
    .formula {
        background: #f0f4f8;
        border-left: 4px solid #4a90e2;
        color: #1a1a2e;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .highlight {
        background: #e3f2fd;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🏭 Модель межотраслевого баланса Леонтьева")
    
    st.markdown("""
    <div class="about-section">
        <h3 class="about-title">📖 О проекте</h3>
        <p>Модель Леонтьева (затраты-выпуск) — экономико-математическая модель, 
        разработанная лауреатом Нобелевской премии Василием Леонтьевым в 1930-х годах.</p>
        <p>Модель позволяет анализировать межотраслевые связи и прогнозировать влияние изменений 
        в одной отрасли на всю экономику.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-section">
        <h3 class="about-title">🔢 Математическая модель</h3>
        <p><strong>Основное уравнение:</strong></p>
        <div class="formula">
            X = A·X + Y
        </div>
        <p>где:</p>
        <ul>
            <li><strong>X</strong> — вектор валового выпуска</li>
            <li><strong>A</strong> — матрица прямых затрат (технологические коэффициенты)</li>
            <li><strong>Y</strong> — вектор конечного спроса</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-section">
        <h3 class="about-title">🔢 Решение модели</h3>
        <div class="formula">
            X = (I - A)<sup>-1</sup> · Y
        </div>
        <p>Матрица <strong class="highlight">L = (I - A)<sup>-1</sup></strong> называется <strong>матрицей Леонтьева</strong> или матрицей полных затрат.</p>
        <p>Элемент <strong class="highlight">l<sub>ij</sub></strong> показывает, сколько нужно произвести в отрасли i, 
        чтобы удовлетворить единицу конечного спроса на продукцию отрасли j.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-section">
        <h3 class="about-title">📊 Что можно анализировать</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px;">
                <strong>🎯 Мультипликаторы</strong><br>
                Показывают, насколько вырастет экономика при увеличении спроса
            </div>
            <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px;">
                <strong>🔗 Сетевые связи</strong><br>
                Выявляют ключевые отрасли и критические зависимости
            </div>
            <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px;">
                <strong>🎯 Сценарный анализ</strong><br>
                Прогнозирует влияние шоков на экономику
            </div>
            <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px;">
                <strong>⚡ Производительность</strong><br>
                Параллельные вычисления для больших матриц
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-section">
        <h3 class="about-title">🌍 Источники данных</h3>
        <p><strong>🇪🇺 Eurostat</strong> — официальные данные Евросоюза (64 отрасли, 2010-2022)</p>
        <p><strong>🌍 EXIOBASE</strong> — глобальная экологическая ММВ-таблица (200+ отраслей, 11 стран)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Информация о текущей сессии
    if data.get('metadata', {}).get('data_loaded'):
        st.markdown("---")
        st.markdown("### 📈 Текущая сессия")
        
        meta = data['metadata']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Страна", f"{meta.get('country_code', 'N/A')}")
            st.metric("Год", meta.get('year', 'N/A'))
        with col2:
            st.metric("Отраслей", meta.get('n_industries', 0))
            st.metric("Источник", meta.get('source_name', 'Eurostat'))
        with col3:
            st.metric("Средний мультипликатор", f"{meta.get('avg_output_multiplier', 0):.3f}")
            st.metric("Время расчёта", f"{meta.get('computation_time', 0):.2f} сек")
    
    st.markdown("""
    <div class="about-section" style="text-align: center;">
        <p>© 2025 | МГТУ им. Н.Э. Баумана | Кафедра Операционных систем</p>
    </div>
    """, unsafe_allow_html=True)