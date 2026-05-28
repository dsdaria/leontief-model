   
"""
sparse_solver.py - Страница разреженного решателя СЛАУ
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
import psutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cpp_bridge import get_cg_solver, is_cpp_available


def generate_random_matrix(n: int, density: float = 0.1):
    """Генерация случайной разреженной матрицы"""
    np.random.seed(42)
    
    from scipy.sparse import random as sparse_random
    
    A_sparse = sparse_random(n, n, density=density, format='csr', random_state=42)
    A = A_sparse.toarray() * 0.05
    
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 0.05
    
    col_sums = np.abs(A).sum(axis=0)
    max_sum = col_sums.max()
    if max_sum >= 0.95:
        A = A * 0.8 / max_sum
    
    I_minus_A = np.eye(n) - A
    Y = np.random.randn(n) * 50000
    
    return {
        'A': I_minus_A.astype(np.float32),
        'b': Y.astype(np.float32),
        'n': n,
        'density': density,
        'nnz': np.count_nonzero(I_minus_A)
    }


def solve_cpp(A, b, num_threads, max_iter=500, tol=1e-6):
    """C++ параллельное решение"""
    solver = get_cg_solver()
    start = time.perf_counter()
    x, iterations, residual, elapsed, converged = solver.solve_cg(
        A.astype(np.float64), b.astype(np.float64), tol, max_iter, num_threads
    )
    return x, iterations, residual, elapsed, converged


def render_sparse_solver(data):
    st.markdown("## 🔬 Разреженный решатель (CG метод)")
    st.markdown("### Сравнение производительности: 1, 2, 4 потока")
    
    cpp_available = is_cpp_available()
    
    col1, col2 = st.columns(2)
    with col1:
        if cpp_available:
            st.success("✅ C++ решатель ДОСТУПЕН")
        else:
            st.error("❌ C++ решатель НЕ ДОСТУПЕН")
    with col2:
        st.info(f"💻 CPU ядер: {psutil.cpu_count()}")
    
    st.markdown("---")
    
    # ========== ВЫБОР РЕЖИМА ==========
    st.markdown("### 📂 Выберите режим работы")
    
    mode = st.radio(
        "Режим:",
        ["📊 Реальная матрица (из парсинга Eurostat/EXIOBASE)", "🎲 Рандомная матрица (для тестов)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # ========== РЕЖИМ 1: РЕАЛЬНАЯ МАТРИЦА ==========
    if mode == "📊 Реальная матрица (из парсинга Eurostat/EXIOBASE)":
        st.markdown("### 📊 Матрица из текущей модели")
        
        if not data or not data.get('metadata', {}).get('data_loaded'):
            st.warning("⚠️ Модель не загружена! Сначала выберите страну и год в боковой панели.")
            return
        
        A_df = data.get('A')
        if A_df is None:
            st.error("❌ Матрица A не найдена в загруженных данных!")
            return
        
        if isinstance(A_df, pd.DataFrame):
            A = A_df.values
        else:
            A = A_df
        
        n = A.shape[0]
        
        Y = data.get('Y')
        if Y is None:
            Y = np.random.randn(n) * 1000
        else:
            if isinstance(Y, pd.Series):
                Y = Y.values
        
        I_minus_A = np.eye(n) - A
        
        max_iter = st.number_input("Макс. итераций:", min_value=100, max_value=2000, value=500, step=100)
        
        nnz = np.count_nonzero(I_minus_A)
        st.info(f"""
        📊 **Реальная матрица:**
        - Источник: {data['metadata'].get('source_name', 'Eurostat')}
        - Страна: {data['metadata'].get('country_code', '?')}
        - Год: {data['metadata'].get('year', '?')}
        - Размер: {n}×{n}
        - Ненулевых: {nnz:,} ({nnz/(n*n)*100:.2f}%)
        """)
        
        with st.expander("🔍 Предпросмотр матрицы (первые 10×10)"):
            preview_n = min(10, n)
            industries = data.get('industries', [f"ind_{i}" for i in range(n)])
            preview_df = pd.DataFrame(
                I_minus_A[:preview_n, :preview_n],
                index=industries[:preview_n],
                columns=industries[:preview_n]
            )
            st.dataframe(preview_df.round(4), use_container_width=True)
        
        st.session_state['sparse_A'] = I_minus_A.astype(np.float32)
        st.session_state['sparse_b'] = Y.astype(np.float32)
        st.session_state['sparse_n'] = n
        st.session_state['sparse_nnz'] = nnz
        st.session_state['matrix_ready'] = True
        st.session_state['matrix_mode'] = "real"
    
    # ========== РЕЖИМ 2: РАНДОМНАЯ МАТРИЦА ==========
    else:
        st.markdown("### 🎲 Параметры рандомной матрицы")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n = st.number_input("Размер матрицы n:", min_value=100, max_value=30000, value=500, step=100)
        with col2:
            density_percent = st.slider("Плотность (%):", min_value=1.0, max_value=30.0, value=10.0, step=1.0)
            density = density_percent / 100.0
        with col3:
            max_iter = st.number_input("Макс. итераций:", min_value=100, max_value=2000, value=500, step=100)
        
        if st.button("🎲 СГЕНЕРИРОВАТЬ МАТРИЦУ", type="primary", use_container_width=True):
            with st.spinner(f"Генерация матрицы {n}×{n}..."):
                mat = generate_random_matrix(int(n), density)
                st.session_state['sparse_A'] = mat['A']
                st.session_state['sparse_b'] = mat['b']
                st.session_state['sparse_n'] = mat['n']
                st.session_state['sparse_nnz'] = mat['nnz']
                st.session_state['matrix_ready'] = True
                st.session_state['matrix_mode'] = "random"
            
            st.success(f"✅ Матрица {n}×{n} сгенерирована!")
            st.info(f"📊 Ненулевых: {mat['nnz']:,} (плотность {mat['nnz']/(n*n)*100:.1f}%)")
        
        if st.session_state.get('matrix_ready') and st.session_state.get('matrix_mode') == "random":
            st.info(f"📊 Текущая матрица: {st.session_state['sparse_n']}×{st.session_state['sparse_n']}, ненулевых: {st.session_state['sparse_nnz']:,}")
    
    st.markdown("---")
    
    if not st.session_state.get('matrix_ready', False):
        if mode == "📊 Реальная матрица (из парсинга Eurostat/EXIOBASE)":
            st.info("👆 Сначала загрузите модель в боковой панели")
        else:
            st.info("👆 Сначала сгенерируйте матрицу")
        return
    
    A = st.session_state['sparse_A']
    b = st.session_state['sparse_b']
    size = st.session_state['sparse_n']
    
    # ========== ВИЗУАЛИЗАЦИЯ РАБОТЫ ПОТОКОВ ==========
    st.markdown("### 🧵 Как потоки распределяют работу")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: #e8f5e9; padding: 1rem; border-radius: 10px; text-align: center;">
            <span style="font-size: 2rem;">🟢</span>
            <h4>1 поток</h4>
            <p style="font-size: 0.8rem;">Обрабатывает ВСЕ строки<br>1 → {size}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rows_1 = size // 2
        rows_2 = size - rows_1
        st.markdown(f"""
        <div style="background: #fff3e0; padding: 1rem; border-radius: 10px; text-align: center;">
            <span style="font-size: 2rem;">🟠🟠</span>
            <h4>2 потока</h4>
            <p style="font-size: 0.8rem;">Поток 1: строки 1 → {rows_1}<br>Поток 2: строки {rows_1+1} → {size}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        rows_4 = size // 4
        st.markdown(f"""
        <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; text-align: center;">
            <span style="font-size: 2rem;">🔵🔵🔵🔵</span>
            <h4>4 потока</h4>
            <p style="font-size: 0.8rem;">Поток 1: строки 1 → {rows_4}<br>Поток 2: {rows_4+1} → {2*rows_4}<br>Поток 3: {2*rows_4+1} → {3*rows_4}<br>Поток 4: {3*rows_4+1} → {size}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== ТЕСТ 1, 2, 4 ПОТОКА ==========
    if st.button("🚀 ЗАПУСТИТЬ ТЕСТ (1, 2, 4 потока)", type="primary", use_container_width=True):
        if not cpp_available:
            st.error("C++ решатель недоступен!")
            return
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        threads_list = [1, 2, 4]
        
        for i, threads in enumerate(threads_list):
            status_text.text(f"Тестирование с {threads} поток(ов)...")
            progress_bar.progress(0.1 + 0.9 * (i / len(threads_list)))
            
            times = []
            iterations_list = []
            
            for run in range(2):
                x, iters, resid, elapsed, conv = solve_cpp(A, b, threads, max_iter=max_iter, tol=1e-6)
                times.append(elapsed)
                iterations_list.append(iters)
            
            avg_time = np.mean(times)
            avg_iterations = int(np.mean(iterations_list))
            
            results.append({
                'threads': threads,
                'time': avg_time,
                'iterations': avg_iterations,
            })
            
            status_text.text(f"✓ {threads} поток(ов): {avg_time:.4f} сек ({avg_iterations} итераций)")
        
        progress_bar.empty()
        status_text.empty()
        
        base_time = results[0]['time']
        for r in results:
            r['speedup'] = base_time / r['time']
            r['efficiency'] = (r['speedup'] / r['threads']) * 100
        
        st.session_state['benchmark_results'] = results
        st.success("✅ Тестирование завершено!")
    
    st.markdown("---")
    
    # ========== РЕЗУЛЬТАТЫ ==========
    if st.session_state.get('benchmark_results'):
        results = st.session_state['benchmark_results']
        
        # Таблица результатов
        df = pd.DataFrame([{
            'Потоков': r['threads'],
            'Время (сек)': f"{r['time']:.4f}",
            'Итерации': r['iterations'],
            'Ускорение': f"{r['speedup']:.2f}x",
            'Эффективность': f"{r['efficiency']:.1f}%"
        } for r in results])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # ========== ГРАФИК ВРЕМЕНИ ==========
        st.markdown("### ⏱️ Сравнение времени выполнения")
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=[r['threads'] for r in results],
            y=[r['time'] for r in results],
            text=[f"{r['time']:.3f} сек" for r in results],
            textposition='outside',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
        ))
        fig_time.update_layout(
            title=f"Время решения (матрица {size}×{size})",
            xaxis_title="Количество потоков",
            yaxis_title="Время (секунды)",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # ========== ГРАФИК УСКОРЕНИЯ ==========
        st.markdown("### ⚡ Ускорение относительно 1 потока")
        
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Scatter(
            x=[r['threads'] for r in results],
            y=[r['speedup'] for r in results],
            mode='lines+markers',
            name='Реальное ускорение',
            line=dict(color='#28a745', width=3),
            marker=dict(size=12, color='#28a745')
        ))
        fig_speed.add_trace(go.Scatter(
            x=[1, 2, 4],
            y=[1, 2, 4],
            mode='lines',
            name='Идеальное ускорение',
            line=dict(color='red', width=2, dash='dash')
        ))
        fig_speed.update_layout(
            title="Ускорение (чем ближе к идеальной линии, тем лучше)",
            xaxis_title="Количество потоков",
            yaxis_title="Ускорение (x)",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_speed, use_container_width=True)
        
        # ========== ГРАФИК ЭФФЕКТИВНОСТИ ==========
        st.markdown("### 📊 Эффективность параллелизации")
        
        fig_eff = go.Figure()
        fig_eff.add_trace(go.Bar(
            x=[r['threads'] for r in results if r['threads'] > 1],
            y=[r['efficiency'] for r in results if r['threads'] > 1],
            text=[f"{r['efficiency']:.1f}%" for r in results if r['threads'] > 1],
            textposition='outside',
            marker_color=['#ff9800', '#ff9800']
        ))
        fig_eff.update_layout(
            title="Эффективность использования потоков",
            xaxis_title="Количество потоков",
            yaxis_title="Эффективность (%)",
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig_eff, use_container_width=True)
        
        # ========== ДЕТАЛЬНЫЙ ВЫВОД ПО КАЖДОМУ ПОТОКУ ==========
        st.markdown("### 📝 Что делал каждый поток")
        
        for r in results:
            if r['threads'] == 1:
                st.info(f"""
                **🟢 1 поток (последовательный режим):**
                - Время: **{r['time']:.4f} сек**
                - Итераций CG: {r['iterations']}
                - Обработал строки: **1 → {size}** (все строки)
                - Это базовый уровень для сравнения
                """)
            elif r['threads'] == 2:
                rows_per = size // 2
                st.info(f"""
                **🟠 2 потока (параллельный режим):**
                - Время: **{r['time']:.4f} сек**
                - Итераций CG: {r['iterations']}
                - Ускорение: **{r['speedup']:.2f}x**
                - Эффективность: **{r['efficiency']:.1f}%**
                - **Поток 1:** обработал строки **1 → {rows_per}**
                - **Поток 2:** обработал строки **{rows_per + 1} → {size}**
                """)
            else:
                rows_per = size // 4
                st.info(f"""
                **🔵 4 потока (параллельный режим):**
                - Время: **{r['time']:.4f} сек**
                - Итераций CG: {r['iterations']}
                - Ускорение: **{r['speedup']:.2f}x**
                - Эффективность: **{r['efficiency']:.1f}%**
                - **Поток 1:** обработал строки **1 → {rows_per}**
                - **Поток 2:** обработал строки **{rows_per + 1} → {2 * rows_per}**
                - **Поток 3:** обработал строки **{2 * rows_per + 1} → {3 * rows_per}**
                - **Поток 4:** обработал строки **{3 * rows_per + 1} → {size}**
                """)
        
        best = max(results, key=lambda x: x['speedup'])
        st.success(f"🏆 **Лучший результат:** {best['threads']} потока — ускорение {best['speedup']:.2f}x, время {best['time']:.4f} сек")
    
    # ========== МНОГОВАРИАНТНЫЕ РАСЧЕТЫ ==========
    st.markdown("---")
    st.markdown("## 📊 МНОГОВАРИАНТНЫЕ РАСЧЕТЫ (разные Y)")
    
    col1, col2 = st.columns(2)
    with col1:
        n_scenarios = st.number_input("Количество сценариев:", min_value=5, max_value=30, value=10, step=5, key="batch_scenarios")
    with col2:
        batch_threads = st.number_input("Потоков для batch:", min_value=1, max_value=psutil.cpu_count(), value=4, step=1, key="batch_threads_input")
    
    if st.button("🚀 ЗАПУСТИТЬ МНОГОВАРИАНТНЫЙ РАСЧЕТ", use_container_width=True, key="batch_btn"):
        if not cpp_available:
            st.error("C++ решатель недоступен!")
        else:
            A_mat = st.session_state['sparse_A']
            n_size = st.session_state['sparse_n']
            base_Y = st.session_state['sparse_b']
            solver = get_cg_solver()
            
            np.random.seed(42)
            B = np.zeros((int(n_scenarios), n_size))
            for i in range(int(n_scenarios)):
                variation = 1 + (np.random.randn(n_size) * 0.2)
                B[i] = base_Y * variation
            
            with st.spinner("Последовательное решение..."):
                seq_times = []
                for i in range(int(n_scenarios)):
                    _, _, _, elapsed, _ = solver.solve_cg(A_mat.astype(np.float64), B[i].astype(np.float64), 1e-6, 500, 1)
                    seq_times.append(elapsed)
                seq_total = sum(seq_times)
            
            with st.spinner(f"Batch решение ({batch_threads} потоков)..."):
                X_batch, batch_time = solver.solve_batch_cg(A_mat.astype(np.float64), B.astype(np.float64), 1e-6, 500, batch_threads)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Последовательно", f"{seq_total:.3f} сек")
            with col_b:
                st.metric(f"Batch ({batch_threads} потоков)", f"{batch_time:.3f} сек")
            with col_c:
                speedup = seq_total / batch_time
                st.metric("Ускорение", f"{speedup:.2f}x")
            
            fig_batch = go.Figure()
            fig_batch.add_trace(go.Bar(
                x=['Последовательно', f'Batch ({batch_threads} потоков)'],
                y=[seq_total, batch_time],
                text=[f'{seq_total:.2f} сек', f'{batch_time:.2f} сек'],
                textposition='auto',
                marker_color=['#1f77b4', '#2ca02c']
            ))
            fig_batch.update_layout(title=f"{n_scenarios} сценариев", height=350)
            st.plotly_chart(fig_batch, use_container_width=True)
            
            st.success(f"✅ Ускорение {speedup:.2f}x при решении {n_scenarios} систем!")
    
    # Кнопка сброса
    st.markdown("---")
    if st.button("🔄 СБРОСИТЬ РЕЗУЛЬТАТЫ", use_container_width=True):
        if 'benchmark_results' in st.session_state:
            del st.session_state['benchmark_results']
        if 'matrix_ready' in st.session_state:
            del st.session_state['matrix_ready']
        st.rerun()