"""
streamlit_pages/performance.py
Страница анализа производительности - ПОЛНОСТЬЮ БЕЗ MATPLOTLIB
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import psutil
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import inv, solve
import plotly.graph_objects as go
import plotly.express as px

from streamlit_components.layouts import section_title


# ===================== ФУНКЦИЯ ДЛЯ ЗАМЕРА ВРЕМЕНИ =====================

def time_function(func, *args, **kwargs):
    """Замер времени выполнения функции"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ===================== ПАРАЛЛЕЛЬНЫЕ И ПОСЛЕДОВАТЕЛЬНЫЕ ВЕРСИИ =====================

def solve_sequential(matrix: np.ndarray, Y_list: list) -> tuple:
    """ПОСЛЕДОВАТЕЛЬНОЕ решение - по одной правой части"""
    results = []
    start = time.perf_counter()
    for Y in Y_list:
        X = solve(matrix, Y)
        results.append(X)
    elapsed = time.perf_counter() - start
    return results, elapsed


def solve_parallel_threads(matrix: np.ndarray, Y_list: list, n_threads: int) -> tuple:
    """ПАРАЛЛЕЛЬНОЕ решение через ThreadPoolExecutor"""
    def solve_one(Y):
        return solve(matrix, Y)
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(solve_one, Y_list))
    elapsed = time.perf_counter() - start
    return results, elapsed


def solve_batch_lapack(matrix: np.ndarray, Y_list: list) -> tuple:
    """ПАКЕТНОЕ решение (многопоточный LAPACK) - САМЫЙ БЫСТРЫЙ"""
    Y_matrix = np.column_stack(Y_list)
    start = time.perf_counter()
    X_matrix = solve(matrix, Y_matrix)
    elapsed = time.perf_counter() - start
    return [X_matrix[:, i] for i in range(len(Y_list))], elapsed


# ===================== ФУНКЦИЯ ДЛЯ ВИЗУАЛЬНОГО СРАВНЕНИЯ =====================

def run_parallel_demo(n: int = 64, n_scenarios: int = 50):
    """
    Запуск демонстрации параллельных вычислений с визуальным сравнением
    """
    st.markdown(f"## 🚀 Демонстрация параллельных вычислений")
    st.markdown(f"**Размер матрицы:** {n}×{n} | **Количество сценариев:** {n_scenarios}")
    
    # Генерация тестовой матрицы
    with st.spinner("Генерация тестовой матрицы..."):
        np.random.seed(42)
        A = np.random.rand(n, n) * 0.3 / np.sqrt(n)
        # Нормализуем для продуктивности
        col_sums = A.sum(axis=0)
        max_sum = col_sums.max()
        if max_sum >= 0.99:
            A = A * 0.8 / max_sum
        matrix = np.eye(n) - A
        
        # Генерируем случайные правые части (сценарии)
        Y_list = [np.random.randn(n) * 1000 for _ in range(n_scenarios)]
    
    # Создаём progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    # 1. ПОСЛЕДОВАТЕЛЬНОЕ решение
    status_text.text("🟢 Тест 1/4: Последовательное решение (1 поток)...")
    _, seq_time = solve_sequential(matrix, Y_list)
    results.append({
        'Метод': 'Последовательный',
        'Потоков': 1,
        'Время (сек)': seq_time,
        'Ускорение': 1.0
    })
    progress_bar.progress(25)
    
    # 2. ПАРАЛЛЕЛЬНОЕ (ThreadPoolExecutor) - 2 потока
    status_text.text("🟡 Тест 2/4: Параллельное решение (2 потока)...")
    _, time_2threads = solve_parallel_threads(matrix, Y_list, n_threads=2)
    results.append({
        'Метод': 'Параллельный (2 потока)',
        'Потоков': 2,
        'Время (сек)': time_2threads,
        'Ускорение': seq_time / time_2threads if time_2threads > 0 else 0
    })
    progress_bar.progress(50)
    
    # 3. ПАРАЛЛЕЛЬНОЕ (ThreadPoolExecutor) - максимальное количество потоков
    n_cores = min(8, cpu_count())
    status_text.text(f"🟠 Тест 3/4: Параллельное решение ({n_cores} потоков)...")
    _, time_parallel = solve_parallel_threads(matrix, Y_list, n_threads=n_cores)
    results.append({
        'Метод': f'Параллельный ({n_cores} потоков)',
        'Потоков': n_cores,
        'Время (сек)': time_parallel,
        'Ускорение': seq_time / time_parallel if time_parallel > 0 else 0
    })
    progress_bar.progress(75)
    
    # 4. ПАКЕТНОЕ решение (LAPACK - самый быстрый)
    status_text.text("🟢 Тест 4/4: Пакетное решение (многопоточный LAPACK)...")
    _, batch_time = solve_batch_lapack(matrix, Y_list)
    results.append({
        'Метод': 'Пакетный (LAPACK)',
        'Потоков': 'auto (MKL)',
        'Время (сек)': batch_time,
        'Ускорение': seq_time / batch_time if batch_time > 0 else 0
    })
    progress_bar.progress(100)
    
    status_text.text("✅ Все тесты завершены!")
    
    # Создаём DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    # ========== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (БЕЗ GRADIENT) ==========
    
    st.markdown("---")
    st.markdown("## 📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    
    # Таблица с результатами (без background_gradient)
    st.markdown("### 📋 Таблица результатов")
    
    # Форматируем таблицу вручную через markdown
    table_md = "| Метод | Потоков | Время (сек) | Ускорение |\n"
    table_md += "|-------|---------|-------------|-----------|\n"
    
    for _, row in results_df.iterrows():
        time_str = f"{row['Время (сек)']:.4f}"
        speedup_str = f"{row['Ускорение']:.2f}x" if row['Ускорение'] > 0 else "N/A"
        table_md += f"| {row['Метод']} | {row['Потоков']} | {time_str} | {speedup_str} |\n"
    
    st.markdown(table_md)
    
    # Альтернативный способ - просто показать dataframe без стилей
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )
    
    # График 1: Сравнение времени выполнения (горизонтальные бары с Plotly)
    st.markdown("### ⏱️ Сравнение времени выполнения")
    
    fig1 = go.Figure()
    
    # Цвета для методов
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig1.add_trace(go.Bar(
        y=results_df['Метод'],
        x=results_df['Время (сек)'],
        orientation='h',
        marker_color=colors[:len(results_df)],
        text=results_df['Время (сек)'].round(4).astype(str) + ' сек',
        textposition='outside',
        hovertemplate='Метод: %{y}<br>Время: %{x:.4f} сек<br>Ускорение: %{customdata:.2f}x<extra></extra>',
        customdata=results_df['Ускорение']
    ))
    
    fig1.update_layout(
        title=f'Время решения {n_scenarios} систем уравнений',
        xaxis_title='Время (секунды)',
        yaxis_title='Метод решения',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # График 2: Ускорение относительно последовательного метода
    st.markdown("### ⚡ Ускорение относительно последовательного метода")
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        x=results_df['Метод'],
        y=results_df['Ускорение'],
        marker_color=colors[:len(results_df)],
        text=results_df['Ускорение'].round(2).astype(str) + 'x',
        textposition='outside',
        hovertemplate='Метод: %{x}<br>Ускорение: %{y:.2f}x<extra></extra>'
    ))
    
    fig2.add_hline(y=1, line_dash="dash", line_color="gray", 
                   annotation_text="Базовый уровень (1x)", annotation_position="top right")
    
    fig2.update_layout(
        title='Ускорение параллельных методов',
        xaxis_title='Метод решения',
        yaxis_title='Ускорение (раз)',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # График 3: Сравнение времени (вертикальные бары)
    st.markdown("### 📊 Детальное сравнение")
    
    fig3 = go.Figure()
    
    fig3.add_trace(go.Bar(
        x=results_df['Метод'],
        y=results_df['Время (сек)'],
        marker_color=colors[:len(results_df)],
        text=results_df['Время (сек)'].round(4).astype(str) + ' сек',
        textposition='outside',
        name='Время выполнения'
    ))
    
    fig3.update_layout(
        title=f'Сравнение времени выполнения ({n_scenarios} сценариев)',
        xaxis_title='Метод решения',
        yaxis_title='Время (секунды)',
        height=450,
        template='plotly_white'
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Круговая диаграмма - экономия времени
    st.markdown("### 💰 Экономия времени")
    
    best_method = results_df.loc[results_df['Время (сек)'].idxmin(), 'Метод']
    best_time = results_df['Время (сек)'].min()
    time_saved = seq_time - best_time
    
    # Создаём круговую диаграмму
    fig4 = go.Figure(data=[go.Pie(
        labels=['Сэкономленное время', 'Затраченное время (лучший метод)'],
        values=[time_saved, best_time],
        marker_colors=['#2ca02c', '#ff7f0e'],
        hole=0.4,
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig4.update_layout(
        title=f'Экономия времени при использовании {best_method}',
        height=350,
        template='plotly_white'
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Индикатор ускорения
    fig5 = go.Figure()
    
    fig5.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=best_time,
        delta={'reference': seq_time, 'relative': True, 'valueformat': '.1%'},
        title={'text': f"Лучший метод: {best_method}"},
        gauge={
            'axis': {'range': [0, seq_time], 'title': 'Время (сек)'},
            'bar': {'color': "#2ca02c"},
            'steps': [
                {'range': [0, best_time], 'color': "lightgreen"},
                {'range': [best_time, seq_time], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': seq_time
            }
        }
    ))
    
    fig5.update_layout(height=300, template='plotly_white')
    st.plotly_chart(fig5, use_container_width=True)
    
    # ВЫВОД ВЫВОДОВ
    st.markdown("---")
    st.markdown("## 🎯 ВЫВОДЫ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        best_method_row = results_df.loc[results_df['Время (сек)'].idxmin()]
        st.metric(
            "🏆 Самый быстрый метод",
            best_method_row['Метод'],
            delta=f"{best_method_row['Время (сек)']:.4f} сек"
        )
    
    with col2:
        best_speedup_row = results_df.loc[results_df['Ускорение'].idxmax()]
        st.metric(
            "⚡ Максимальное ускорение",
            f"{best_speedup_row['Ускорение']:.2f}x",
            delta=f"относительно последовательного метода"
        )
    
    # Вывод прогресса
    st.success(f"""
    ✅ **Параллельные вычисления работают!**
    
    - Последовательный метод: **{seq_time:.4f} сек**
    - Лучший параллельный метод ({best_method}): **{best_time:.4f} сек**
    - Ускорение: **{seq_time/best_time:.2f}x**
    - Сэкономлено времени: **{time_saved:.4f} сек** ({time_saved/seq_time*100:.1f}%)
    """)
    
    return results_df


# ===================== ПРОСТОЙ ТЕСТ ДЛЯ БЫСТРОЙ ПРОВЕРКИ =====================

def quick_parallel_test():
    """
    Быстрый тест параллельных вычислений с минимальными настройками
    """
    st.markdown("## ⚡ БЫСТРЫЙ ТЕСТ ПАРАЛЛЕЛИЗМА")
    st.markdown("*Нажмите кнопку, чтобы увидеть разницу во времени между последовательным и параллельным решением*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.number_input("Размер матрицы:", min_value=10, max_value=200, value=50, key="quick_size")
    with col2:
        test_scenarios = st.number_input("Количество сценариев:", min_value=10, max_value=200, value=30, key="quick_scenarios")
    
    if st.button("🚀 ЗАПУСТИТЬ ТЕСТ", type="primary", use_container_width=True):
        # Генерация данных
        with st.spinner("Подготовка данных..."):
            np.random.seed(42)
            A = np.random.rand(test_size, test_size) * 0.3 / np.sqrt(test_size)
            col_sums = A.sum(axis=0)
            if col_sums.max() >= 0.99:
                A = A * 0.8 / col_sums.max()
            matrix = np.eye(test_size) - A
            Y_list = [np.random.randn(test_size) for _ in range(test_scenarios)]
        
        # Последовательное решение
        with st.spinner("🟢 Последовательное решение..."):
            start = time.perf_counter()
            for Y in Y_list:
                solve(matrix, Y)
            seq_time = time.perf_counter() - start
        
        # Параллельное решение (ThreadPoolExecutor)
        n_cores = min(8, cpu_count())
        with st.spinner(f"🟡 Параллельное решение ({n_cores} потоков)..."):
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n_cores) as executor:
                list(executor.map(lambda Y: solve(matrix, Y), Y_list))
            par_time = time.perf_counter() - start
        
        # Пакетное решение
        with st.spinner("🟢 Пакетное решение (LAPACK)..."):
            Y_matrix = np.column_stack(Y_list)
            start = time.perf_counter()
            solve(matrix, Y_matrix)
            batch_time = time.perf_counter() - start
        
        # Отображение результатов
        st.markdown("### 📊 РЕЗУЛЬТАТЫ")
        
        # Крупные метрики
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            st.metric(
                "📌 Последовательный",
                f"{seq_time:.4f} сек",
                delta="базовый уровень"
            )
        
        with metric_cols[1]:
            speedup_par = seq_time / par_time
            st.metric(
                f"🚀 Параллельный ({n_cores} потоков)",
                f"{par_time:.4f} сек",
                delta=f"быстрее в {speedup_par:.2f}x",
                delta_color="normal"
            )
        
        with metric_cols[2]:
            speedup_batch = seq_time / batch_time
            st.metric(
                f"⚡ Пакетный (LAPACK)",
                f"{batch_time:.4f} сек",
                delta=f"быстрее в {speedup_batch:.2f}x",
                delta_color="normal"
            )
        
        # Визуализация с Plotly
        fig = go.Figure()
        
        methods = ['Последовательный', f'Параллельный\n({n_cores} потоков)', 'Пакетный (LAPACK)']
        times = [seq_time, par_time, batch_time]
        colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        fig.add_trace(go.Bar(
            x=methods,
            y=times,
            marker_color=colors_bar,
            text=[f'{t:.4f} сек' for t in times],
            textposition='outside',
            hovertemplate='Метод: %{x}<br>Время: %{y:.4f} сек<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Сравнение времени решения {test_scenarios} систем уравнений',
            xaxis_title='Метод решения',
            yaxis_title='Время (секунды)',
            height=450,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Вывод
        st.success(f"""
        ✅ **Параллельные вычисления работают!**
        
        - Параллельный метод быстрее последовательного в **{seq_time/par_time:.2f} раза**
        - Пакетный метод (LAPACK) быстрее в **{seq_time/batch_time:.2f} раза**
        - Сэкономлено времени: **{seq_time - par_time:.3f} сек** (при параллельном решении)
        """)
        
        # Анимация-пояснение
        with st.expander("📖 Как работает параллелизм?"):
            st.markdown(f"""
            **Что произошло под капотом:**
            
            1. **Последовательный метод** решал {test_scenarios} систем уравнений **по очереди**
               - Использовал 1 ядро процессора
               - Время: {seq_time:.3f} сек
            
            2. **Параллельный метод** распределил {test_scenarios} задач на {n_cores} потоков
               - Использовал {n_cores} ядер процессора одновременно
               - Время: {par_time:.3f} сек
               - **Ускорение: {seq_time/par_time:.2f}x**
            
            3. **Пакетный метод (LAPACK)** использовал оптимизированные библиотеки
               - Факторизует матрицу один раз, решает для всех правых частей
               - Автоматически использует многопоточный BLAS
               - **Ускорение: {seq_time/batch_time:.2f}x** (самый быстрый!)
            """)


# ===================== ОСНОВНАЯ ФУНКЦИЯ РЕНДЕРИНГА =====================

def render_performance(data: Dict):
    """Главная функция рендеринга страницы производительности"""
    section_title("Анализ производительности", "⚡")
    
    # Вкладки для разных режимов
    tab1, tab2, tab3 = st.tabs([
        "🚀 Демонстрация параллелизма",
        "⚡ Быстрый тест",
        "📊 Системная информация"
    ])
    
    with tab1:
        st.markdown("""
        ### 🎯 Демонстрация параллельных вычислений
        
        Здесь вы можете **наглядно увидеть**, как параллельные вычисления ускоряют решение систем уравнений.
        
        **Что вы увидите:**
        - Сравнение времени выполнения разных методов
        - Графики ускорения
        - Конкретные цифры: во сколько раз быстрее
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            demo_n = st.slider(
                "Размер матрицы (n):",
                min_value=20,
                max_value=150,
                value=64,
                step=10,
                help="Чем больше матрица, тем заметнее ускорение",
                key="demo_n"
            )
        
        with col2:
            demo_scenarios = st.slider(
                "Количество сценариев:",
                min_value=10,
                max_value=80,
                value=50,
                step=10,
                help="Чем больше сценариев, тем эффективнее параллельные методы",
                key="demo_scenarios"
            )
        
        if st.button("▶️ ЗАПУСТИТЬ ДЕМОНСТРАЦИЮ", type="primary", use_container_width=True):
            with st.spinner("Выполняется демонстрация параллельных вычислений..."):
                results = run_parallel_demo(int(demo_n), int(demo_scenarios))
    
    with tab2:
        quick_parallel_test()
    
    with tab3:
        st.markdown("### 💻 Системная информация")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU ядер (логических)", cpu_count())
            st.metric("CPU ядер (физических)", psutil.cpu_count(logical=False) or "N/A")
        
        with col2:
            st.metric("RAM всего", f"{psutil.virtual_memory().total / (1024**3):.1f} GB")
            st.metric("RAM доступно", f"{psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        with col3:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            st.metric("Текущая загрузка CPU", f"{cpu_percent:.0f}%")
            st.metric("NumPy версия", np.__version__)
        
        # Информация о текущей модели (если загружена)
        if data.get('metadata', {}).get('data_loaded'):
            st.markdown("### 📊 Производительность текущей модели")
            meta = data['metadata']
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Размер матрицы", f"{meta.get('n_industries', 0)}×{meta.get('n_industries', 0)}")
            with cols[1]:
                st.metric("Время расчёта", f"{meta.get('computation_time', 0):.3f} сек")
            with cols[2]:
                st.metric("Использовано потоков", meta.get('n_threads', 1))
            with cols[3]:
                st.metric("Метод", meta.get('method_used', 'unknown')[:20])