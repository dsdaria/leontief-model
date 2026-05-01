"""
CSS стили для Streamlit приложения - АДАПТИВНЫЙ ДИЗАЙН
"""

import streamlit as st


def inject_custom_css():
    st.markdown("""
    <style>
        /* ========== ОСНОВНЫЕ СТИЛИ ========== */
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #f0f2f5 100%);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #1a1a2e !important;
        }
        
        p, li, label, .stMarkdown, .stText {
            color: #333333 !important;
        }
        
        /* ========== КНОПКИ ========== */
        .stButton > button {
            background: linear-gradient(135deg, #e8f4f8 0%, #d4eaf7 100%);
            border: 2px solid #4a90e2;
            color: #2c3e50;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(74, 144, 226, 0.3);
        }
        
        /* ========== БОКОВАЯ ПАНЕЛЬ ========== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
            border-right: 1px solid #dee2e6;
        }
        section[data-testid="stSidebar"] * {
            color: #2c3e50 !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
            background: white;
            border: 1px solid #4a90e2;
            color: #4a90e2;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: #4a90e2;
            color: white;
        }
        
        /* ========== ИНФОРМАЦИОННЫЕ БЛОКИ ========== */
        .info-box {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdef5 100%);
            border-left: 4px solid #4a90e2;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .info-box.success {
            border-left-color: #28a745;
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        }
        .info-box.warning {
            border-left-color: #ff9800;
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        }
        
        /* ========== КАРТОЧКИ МЕТРИК ========== */
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid #e0e0e0;
        }
        
        /* ========== ТАБЛИЦЫ ========== */
        .stDataFrame {
            background: white;
            border-radius: 10px;
            overflow-x: auto;
        }
        
        /* ========== ВКЛАДКИ ========== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background: white;
            padding: 0.5rem;
            border-radius: 10px;
            flex-wrap: wrap;
        }
        .stTabs [data-baseweb="tab"] {
            color: #2c3e50;
            font-weight: 500;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            white-space: nowrap;
        }
        .stTabs [aria-selected="true"] {
            color: white;
            background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        }
        
        /* ========== ФОРМУЛЫ ========== */
        .formula {
            background: #f8f9fa;
            border-left: 4px solid #4a90e2;
            color: #1a1a2e;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-family: monospace;
            font-size: 1.2rem;
            margin: 1rem 0;
            overflow-x: auto;
        }
        
        /* ========== ВЫПАДАЮЩИЕ СПИСКИ ========== */
        .stSelectbox div[data-baseweb="select"] {
            background-color: white;
            border-color: #4a90e2;
        }
        
        /* ========== СЛАЙДЕРЫ ========== */
        .stSlider div[data-baseweb="slider"] {
            background-color: #4a90e2;
        }
        
        /* ========== ПРОГРЕСС БАР ========== */
        .stProgress > div > div {
            background-color: #4a90e2;
        }
        
        /* ========== ССЫЛКИ ========== */
        a {
            color: #4a90e2;
        }
        a:hover {
            color: #357abd;
        }
        
        /* ========== АНИМАЦИЯ ЗАГРУЗКИ ========== */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        .stSpinner {
            animation: pulse 1.5s infinite;
        }
        
        /* ========== АДАПТИВНЫЙ ДИЗАЙН ДЛЯ ТЕЛЕФОНОВ ========== */
        @media (max-width: 768px) {
            /* Кнопки на всю ширину */
            .stButton > button {
                width: 100%;
                padding: 0.75rem;
                font-size: 1rem;
            }
            
            /* Колонки в столбик */
            .stColumns {
                flex-direction: column;
            }
            .row-widget.stHorizontal {
                flex-wrap: wrap;
            }
            
            /* Карточки метрик */
            .metric-card {
                margin-bottom: 1rem;
                padding: 0.75rem;
            }
            
            /* Заголовки меньше */
            h1 {
                font-size: 1.5rem !important;
            }
            h2 {
                font-size: 1.3rem !important;
            }
            h3 {
                font-size: 1.1rem !important;
            }
            
            /* Таблицы с прокруткой */
            .stDataFrame {
                font-size: 0.8rem;
            }
            .dataframe {
                font-size: 0.7rem;
            }
            
            /* Вкладки для телефона */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 0.4rem 0.8rem;
                font-size: 0.8rem;
            }
            
            /* Отступы меньше */
            .stMarkdown {
                margin-bottom: 0.5rem;
            }
            
            /* Формулы с прокруткой */
            .formula {
                font-size: 0.9rem;
                padding: 0.75rem;
                overflow-x: auto;
                white-space: nowrap;
            }
            
            /* Информационные блоки */
            .info-box {
                padding: 0.75rem 1rem;
                font-size: 0.9rem;
            }
            
            /* Боковая панель уже */
            section[data-testid="stSidebar"] {
                width: 250px !important;
            }
            
            /* Графики адаптивные */
            .js-plotly-plot {
                width: 100% !important;
            }
            
            /* Скрываем лишние элементы на телефоне */
            .stAlert {
                font-size: 0.8rem;
            }
        }
        
        /* ========== ПЛАНШЕТЫ (между телефоном и ПК) ========== */
        @media (min-width: 769px) and (max-width: 1024px) {
            .stButton > button {
                padding: 0.5rem 0.8rem;
            }
            h1 {
                font-size: 1.8rem !important;
            }
            .metric-card {
                padding: 0.8rem;
            }
        }
        
        /* ========== БОЛЬШИЕ ЭКРАНЫ (ПК) ========== */
        @media (min-width: 1025px) {
            .stButton > button {
                min-width: 120px;
            }
        }
    </style>
    """, unsafe_allow_html=True)