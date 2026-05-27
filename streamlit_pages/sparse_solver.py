"""
sparse_solver.py - Страница разреженного решателя СЛАУ
Сравнение производительности + Многовариантные расчеты
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import psutil

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpp_bridge import get_cg_solver, is_cpp_available


def generate_sparse_test_matrix(n: int, density: float = 0.1):
    """Генерация разреженной матрицы"""
    np.random.seed(42)
    A = np.zeros((n, n))
    n_nonzero = int(n * n * density)
    indices = np.random.choice(n * n, size=min(n_nonzero, n*n), replace=False)
    values = np.random.randn(len(indices)) * 0.05
    
    for idx, val in zip(indices, values):
        i, j = idx // n, idx % n
        A[i, j] = val
    
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 0.05
    
    try:
        eigvals = np.linalg.eigvals(A)
        spectral_radius = max(abs(eigvals))
        if spectral_radius >= 0.99:
            A = A * 0.9 / spectral_radius
    except:
        pass
    
    I_minus_A = np.eye(n) - A
    Y = np.random.randn(n) * 50000
    
    return {
        'A': I_minus_A.astype(np.float64),
        'b': Y.astype(np.float64),
        'n': n,
        'density': density,
        'nnz': np.count_nonzero(I_minus_A)
    }


def solve_cpp(A, b, num_threads, max_iter=2000, tol=1e-6):
    """C++ параллельное решение"""
    solver = get_cg_solver()
    start = time.perf_counter()
    x, iterations, residual, elapsed, converged = solver.solve_cg(
        A, b, tol, max_iter, num_threads
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
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input("Размер матрицы n:", min_value=50, max_value=50000, value=500, step=50)
    
    with col2:
        density = st.slider("Плотность (%):", min_value=0.1, max_value=50.0, value=30.0, step=0.5) / 100.0
    
    with col3:
        max_iter = st.number_input("Макс. итераций:", min_value=500, max_value=5000, value=1500, step=500)
        
    # Кнопка генерации
    if st.button("🎲 СГЕНЕРИРОВАТЬ МАТРИЦУ", type="primary", use_container_width=True):
        with st.spinner(f"Генерация матрицы {n}×{n}..."):
            mat = generate_sparse_test_matrix(int(n), density)
            st.session_state['sparse_A'] = mat['A']
            st.session_state['sparse_b'] = mat['b']
            st.session_state['sparse_n'] = mat['n']
            st.session_state['sparse_nnz'] = mat['nnz']
            st.session_state['matrix_ready'] = True
        
        st.success(f"✅ Матрица {n}×{n} сгенерирована!")
        st.info(f"📊 Ненулевых: {mat['nnz']:,} (плотность {mat['nnz']/(n*n)*100:.1f}%)")
    
    st.markdown("---")
    
    if not st.session_state.get('matrix_ready', False):
        st.info("👆 Сгенерируйте матрицу")
        return
    
    A = st.session_state['sparse_A']
    b = st.session_state['sparse_b']
    size = st.session_state['sparse_n']
    
    st.info(f"📊 Матрица: {size}×{size}, {st.session_state['sparse_nnz']:,} ненулевых")
    
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
    
    # Результаты теста
    if st.session_state.get('benchmark_results'):
        results = st.session_state['benchmark_results']
        
        df = pd.DataFrame([{
            'Потоков': r['threads'],
            'Время (сек)': f"{r['time']:.4f}",
            'Итерации': r['iterations'],
            'Ускорение': f"{r['speedup']:.2f}x",
            'Эффективность': f"{r['efficiency']:.1f}%"
        } for r in results])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=[r['threads'] for r in results],
            y=[r['time'] for r in results],
            text=[f"{r['time']:.3f} сек" for r in results],
            textposition='outside',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
        ))
        fig_time.update_layout(title=f"Время решения (матрица {size}×{size})", height=350)
        st.plotly_chart(fig_time, use_container_width=True)
        
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Scatter(
            x=[r['threads'] for r in results],
            y=[r['speedup'] for r in results],
            mode='lines+markers',
            name='Реальное ускорение',
            line=dict(color='#28a745', width=3),
            marker=dict(size=12)
        ))
        fig_speed.add_trace(go.Scatter(
            x=[1, 2, 4],
            y=[1, 2, 4],
            mode='lines',
            name='Идеальное ускорение',
            line=dict(color='red', width=2, dash='dash')
        ))
        fig_speed.update_layout(title="Ускорение", xaxis_title="Потоков", yaxis_title="Ускорение (x)", height=350)
        st.plotly_chart(fig_speed, use_container_width=True)
        
        best = max(results, key=lambda x: x['speedup'])
        st.success(f"🏆 Лучший результат: {best['threads']} потока — ускорение {best['speedup']:.2f}x")
    
    # ========== МНОГОВАРИАНТНЫЕ РАСЧЕТЫ ==========
    st.markdown("---")
    st.markdown("## 📊 МНОГОВАРИАНТНЫЕ РАСЧЕТЫ (разные Y)")
    st.markdown("*Решение СЛАУ для множества правых частей одновременно*")
    
    col1, col2 = st.columns(2)
    with col1:
        n_scenarios = st.number_input("Количество сценариев:", min_value=5, max_value=100, value=20, step=5, key="batch_scenarios")
    with col2:
        batch_threads = st.number_input("Потоков для batch:", min_value=1, max_value=psutil.cpu_count(), value=4, step=1, key="batch_threads_input")
    
    if st.button("🚀 ЗАПУСТИТЬ МНОГОВАРИАНТНЫЙ РАСЧЕТ", use_container_width=True, key="batch_btn"):
        if not st.session_state.get('matrix_ready', False):
            st.error("Сначала сгенерируйте матрицу!")
        else:
            A = st.session_state['sparse_A']
            n_size = st.session_state['sparse_n']
            solver = get_cg_solver()
            
            # Генерируем разные Y
            np.random.seed(42)
            B = np.random.randn(int(n_scenarios), n_size) * 1000
            
            # Последовательное решение
            with st.spinner("Последовательное решение..."):
                seq_times = []
                for i in range(int(n_scenarios)):
                    _, _, _, elapsed, _ = solver.solve_cg(A, B[i], 1e-6, 1500, 1)
                    seq_times.append(elapsed)
                seq_total = sum(seq_times)
            
            # Параллельное batch решение
            with st.spinner(f"Batch решение ({batch_threads} потоков)..."):
                X_batch, batch_time = solver.solve_batch_cg(A, B, 1e-6, 1500, batch_threads)
            
            # Результаты
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Последовательно", f"{seq_total:.3f} сек")
            with col2:
                st.metric(f"Batch ({batch_threads} потоков)", f"{batch_time:.3f} сек")
            with col3:
                speedup = seq_total / batch_time
                st.metric("Ускорение", f"{speedup:.2f}x")
            
            # График
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