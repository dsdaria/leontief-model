"""
Страница демонстрации производительности параллельных вычислений
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamlit_components.layouts import section_title, info_box


def render_performance(data):
    """Рендеринг страницы производительности"""
    
    section_title("⚡ Производительность параллельных вычислений", "⚡")
    
    st.markdown("""
    <div class="info-box">
        🚀 Эта страница позволяет настраивать и тестировать производительность.<br>
        Изменяйте количество потоков и сразу видите время расчёта!
    </div>
    """, unsafe_allow_html=True)
    
    # ========== НАСТРОЙКИ ПРОИЗВОДИТЕЛЬНОСТИ ==========
    render_performance_settings()
    
    st.markdown("---")
    
    # ========== БЫСТРЫЙ ТЕСТ ПРИ ИЗМЕНЕНИИ ПОТОКОВ ==========
    auto_benchmark_on_thread_change(data)
    
    st.markdown("---")
    
    # Информация о текущей системе
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💻 CPU ядер", str(os.cpu_count()), help="Доступно для параллельных расчётов")
    with col2:
        st.metric("⚙️ Потоков MKL", str(st.session_state.get('threads', 8)), help="Используется Intel MKL")
    with col3:
        method_name = st.session_state.get('method', 'BICGSTAB (итерационный)')
        st.metric("📊 Метод", method_name, help="Выбранный метод решения")
    
    st.markdown("---")
    
    # Информация о текущей модели
    if data.get('metadata', {}).get('data_loaded'):
        meta = data['metadata']
        st.info(f"📊 Текущая модель: **{meta.get('n_industries', 0)} отраслей** (источник: {meta.get('source_name', 'Eurostat')})")
    
    st.markdown("---")
    
    # ========== СРАВНЕНИЕ МЕТОДОВ ==========
    render_method_comparison()
    
    st.markdown("---")
    
    # ========== ТЕКУЩАЯ ПРОИЗВОДИТЕЛЬНОСТЬ ==========
    if data.get('metadata', {}).get('data_loaded'):
        render_current_session_performance(data)


def render_performance_settings():
    """Настройки производительности"""
    
    st.markdown("### ⚙️ Настройки производительности")
    
    # Инициализация session state
    if 'method' not in st.session_state:
        st.session_state.method = "BICGSTAB (итерационный)"
    if 'threads' not in st.session_state:
        st.session_state.threads = 8
    if 'tolerance' not in st.session_state:
        st.session_state.tolerance = 1e-8
    if 'maxiter' not in st.session_state:
        st.session_state.maxiter = 1000
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "📐 Метод решения",
            ["BICGSTAB (итерационный)", "GMRES (итерационный)", "Прямой (inv)"],
            index=["BICGSTAB (итерационный)", "GMRES (итерационный)", "Прямой (inv)"].index(st.session_state.method),
            help="BICGSTAB - самый быстрый для больших матриц (>50 отраслей)"
        )
        
        threads = st.slider(
            "⚙️ Количество потоков", 
            min_value=1, 
            max_value=16, 
            value=st.session_state.threads,
            help="Больше потоков = быстрее расчёт, но больше нагрузка на CPU"
        )
        
    with col2:
        if "итерационный" in method:
            tolerance = st.select_slider(
                "🎯 Точность расчёта (tolerance)",
                options=[1e-6, 1e-8, 1e-10, 1e-12],
                value=st.session_state.tolerance,
                format_func=lambda x: f"{x:.0e}",
                help="Меньше = точнее, но медленнее"
            )
            
            maxiter = st.number_input(
                "🔄 Макс. итераций",
                min_value=100,
                max_value=5000,
                value=st.session_state.maxiter,
                step=100,
                help="Максимальное количество итераций"
            )
        else:
            tolerance = st.session_state.tolerance
            maxiter = st.session_state.maxiter
            st.info("ℹ️ Прямой метод не использует итерации, точность максимальная")
    
    # Сохраняем настройки
    if st.button("💾 Сохранить настройки", use_container_width=True):
        st.session_state.method = method
        st.session_state.threads = threads
        st.session_state.tolerance = tolerance
        st.session_state.maxiter = maxiter
        os.environ['OMP_NUM_THREADS'] = str(threads)
        os.environ['MKL_NUM_THREADS'] = str(threads)
        st.success(f"✅ Настройки сохранены! {threads} потоков")
        st.rerun()
    
    st.markdown(f"""
    <div style="background: #f0f4f8; padding: 0.5rem 1rem; border-radius: 8px; margin-top: 0.5rem;">
        <small>📌 Текущие настройки: <strong>{st.session_state.method}</strong>, 
        {st.session_state.threads} потоков</small>
    </div>
    """, unsafe_allow_html=True)


def auto_benchmark_on_thread_change(data):
    """Автоматический запуск бенчмарка при изменении потоков"""
    
    st.markdown("### ⚡ Быстрый тест производительности")
    st.markdown("Изменяйте ползунок - время расчёта обновляется автоматически!")
    
    # Текущие настройки
    current_threads = st.session_state.get('threads', 8)
    
    # Выбор количества потоков для теста
    test_threads = st.slider(
        "🧪 Количество потоков для теста",
        min_value=1,
        max_value=16,
        value=current_threads,
        key="test_threads_slider",
        help="Выберите количество потоков и нажмите 'Замерить время'"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Замерить время", use_container_width=True, key="measure_btn"):
            with st.spinner(f"🔄 Тестирование с {test_threads} потоками..."):
                result = measure_computation_time(data, test_threads)
                
                if result:
                    # Метрика времени
                    st.metric(
                        label=f"⏱️ Время расчёта",
                        value=f"{result['time']:.3f} сек",
                        delta=f"{result['speedup']:.1f}x"
                    )
                    
                    # Прогресс-бар эффективности
                    efficiency = min(100, (result['speedup'] / test_threads * 100))
                    st.progress(efficiency / 100)
                    st.caption(f"⚡ Эффективность: {efficiency:.1f}%")
                    
                    # Рекомендация
                    if test_threads <= 4 and result['speedup'] < 2:
                        st.info("💡 Увеличьте количество потоков для ускорения")
                    elif test_threads >= 12 and result['speedup'] < test_threads * 0.5:
                        st.warning("⚠️ Слишком много потоков - накладные расходы")
                    else:
                        st.success("✅ Оптимальная конфигурация!")
    
    with col2:
        if st.button("📊 Сравнить все конфигурации", use_container_width=True, key="compare_btn"):
            with st.spinner("🔄 Тестирование всех конфигураций..."):
                results = measure_all_threads(data)
                
                if results:
                    # Таблица
                    df = pd.DataFrame(results)
                    df['time'] = df['time'].round(3)
                    base_time = df['time'].iloc[0]
                    df['speedup'] = (base_time / df['time']).round(2)
                    df.columns = ['Потоков', 'Время (сек)', 'Ускорение (x)']
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # График
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df['Потоков'],
                        y=df['Время (сек)'],
                        name='Время расчёта',
                        marker_color='#4a90e2',
                        text=df['Время (сек)'],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Время расчёта при разном количестве потоков",
                        xaxis_title="Количество потоков",
                        yaxis_title="Время (секунды)",
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Лучший результат
                    best_idx = df['Время (сек)'].idxmin()
                    st.success(f"🏆 Лучший результат: {df.loc[best_idx, 'Потоков']} потоков — {df.loc[best_idx, 'Время (сек)']:.3f} сек")


def measure_computation_time(data, threads):
    """Замер времени расчёта с указанным количеством потоков"""
    try:
        from leontief_model import LeontiefModel
        import numpy as np
        import time
        
        if data.get('A') is not None:
            A_matrix = data['A'].values
            n = len(A_matrix)
            industries = data['industries']
            
            # Создаём тестовую матрицу
            Z = A_matrix * np.random.rand(n, n) * 1000
            X = np.abs(Z).sum(axis=1)
            
            model = LeontiefModel(Z, X, industries)
            model.A = A_matrix
            
            # Устанавливаем потоки
            if hasattr(model, 'thread_manager'):
                model.thread_manager.set_num_threads(threads, verbose=False)
            
            # Замеряем время
            start = time.time()
            L, metrics = model.calculate_leontief_matrix_parallel(
                use_iterative=True,
                n_threads=threads,
                method='bicgstab'
            )
            elapsed = time.time() - start
            
            # Получаем базовое время (для 1 потока)
            if 'base_time_1' not in st.session_state:
                st.session_state.base_time_1 = elapsed * threads
            
            base_time = st.session_state.base_time_1 if st.session_state.base_time_1 > 0 else elapsed
            speedup = base_time / elapsed if elapsed > 0 else 1
            
            return {
                'time': elapsed,
                'speedup': speedup
            }
        else:
            st.warning("Данные модели не загружены")
            return None
            
    except Exception as e:
        st.error(f"Ошибка замера: {e}")
        return None


def measure_all_threads(data):
    """Замер времени для всех конфигураций потоков"""
    results = []
    thread_counts = [1, 2, 4, 8, 12, 16]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, threads in enumerate(thread_counts):
        status_text.text(f"🔄 Тестирование с {threads} потоками...")
        result = measure_computation_time(data, threads)
        if result:
            results.append({
                'threads': threads,
                'time': result['time']
            })
        progress_bar.progress((i + 1) / len(thread_counts))
    
    status_text.empty()
    progress_bar.empty()
    
    return results


def render_method_comparison():
    """Сравнение методов решения"""
    
    st.markdown("### 🔬 Сравнение методов решения")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px;">
            <h4>📐 Прямой метод</h4>
            <p><code>L = inv(I - A)</code></p>
            <ul>
                <li>✅ Максимальная точность</li>
                <li>✅ Стабильный</li>
                <li>❌ Медленный для n>100</li>
                <li>❌ Много памяти O(n³)</li>
            </ul>
            <p><strong>Лучше для:</strong> n &lt; 50</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px;">
            <h4>🔄 Итерационный метод</h4>
            <p><code>BICGSTAB / GMRES</code></p>
            <ul>
                <li>✅ Очень быстрый для больших матриц</li>
                <li>✅ Мало памяти O(n²)</li>
                <li>✅ Параллелизуется</li>
                <li>⚠️ Точность настраивается</li>
            </ul>
            <p><strong>Лучше для:</strong> n &gt; 50</p>
        </div>
        """, unsafe_allow_html=True)


def render_current_session_performance(data):
    """Производительность текущей сессии"""
    
    st.markdown("### 📊 Производительность текущей сессии")
    
    meta = data['metadata']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Размер матрицы", f"{meta.get('n_industries', 0)}×{meta.get('n_industries', 0)}")
    with col2:
        t = meta.get('computation_time', 0)
        st.metric("Время расчёта", f"{t:.2f} сек")
    with col3:
        st.metric("Использовано потоков", meta.get('n_threads', st.session_state.get('threads', 8)))
    with col4:
        method = meta.get('method_used', 'unknown')
        if "bicgstab" in str(method).lower():
            method_display = "BICGSTAB"
        elif "gmres" in str(method).lower():
            method_display = "GMRES"
        else:
            method_display = "Прямой метод"
        st.metric("Последний метод", method_display)
    
    memory_used = meta.get('memory_usage_mb', 0)
    if memory_used > 0:
        st.caption(f"💾 Использовано памяти: {memory_used:.1f} MB")
    
    cond = meta.get('condition_number', 0)
    if cond:
        if cond < 1e4:
            st.success(f"✅ Число обусловленности: {cond:.2e} — отличное")
        elif cond < 1e8:
            st.warning(f"⚠️ Число обусловленности: {cond:.2e} — удовлетворительное")
        else:
            st.error(f"❌ Число обусловленности: {cond:.2e} — требуется внимание")