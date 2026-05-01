"""
Клиент для подключения к удаленному решателю
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
from typing import Dict, Optional

# ========== ВАЖНО: URL для облачного деплоя ==========
# Для локальной разработки: http://localhost:5000
# Для облака: читаем из переменной окружения BACKEND_URL
REMOTE_SOLVER_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
USE_REMOTE_SOLVER = True

def check_remote_solver() -> bool:
    """Проверка доступности удаленного решателя"""
    try:
        response = requests.get(f"{REMOTE_SOLVER_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.sidebar.success(f"🟢 Удалённый решатель: {data.get('status', 'online')}")
            return True
    except Exception as e:
        st.sidebar.error(f"🔴 Удалённый решатель недоступен")
    return False

@st.cache_data(ttl=3600, show_spinner="🔄 Отправка запроса на удалённый сервер...")
def load_from_remote_solver(country: str, year: int, source: str, 
                            threads: int = 8, use_iterative: bool = True, 
                            method: str = "bicgstab", tolerance: float = 1e-8, 
                            maxiter: int = 1000) -> Optional[Dict]:
    """Загрузка данных с удаленного сервера с учётом настроек"""
    try:
        payload = {
            'country': country,
            'year': year,
            'source': source,
            'threads': threads,
            'use_iterative': use_iterative,
            'method': method,
            'tolerance': tolerance,
            'maxiter': maxiter
        }
        
        st.info(f"📤 Отправка запроса на {REMOTE_SOLVER_URL}: {threads} потоков")
        
        response = requests.post(
            f"{REMOTE_SOLVER_URL}/api/compute",
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                data = result['data']
                industries = data['industries']
                
                A = pd.DataFrame(data['A'], index=industries, columns=industries)
                multipliers_df = pd.DataFrame(data['multipliers'])
                
                n_connections = int((A.values > 0).sum().sum())
                avg_multiplier = float(multipliers_df['Мультипликатор_выпуска'].mean()) if 'Мультипликатор_выпуска' in multipliers_df.columns else 0.0
                max_multiplier = float(multipliers_df['Мультипликатор_выпуска'].max()) if 'Мультипликатор_выпуска' in multipliers_df.columns else 0.0
                
                scenarios = {}
                for name, scenario_data in data.get('scenarios', {}).items():
                    scenarios[name] = pd.DataFrame(scenario_data)
                
                st.success(f"✅ Ответ получен: {data['n_industries']} отраслей, {threads} потоков")
                
                return {
                    'A': A,
                    'L': pd.DataFrame(data['L'], index=industries, columns=industries),
                    'multipliers': multipliers_df,
                    'scenarios': scenarios,
                    'industries': industries,
                    'metadata': {
                        'data_loaded': True,
                        'n_industries': data['n_industries'],
                        'n_connections': n_connections,
                        'avg_output_multiplier': avg_multiplier,
                        'max_output_multiplier': max_multiplier,
                        'source': source,
                        'source_name': '🌍 EXIOBASE' if source == 'exiobase' else '🇪🇺 Eurostat',
                        'country_code': country,
                        'year': year,
                        'computation_time': data['metadata'].get('computation_time', 0),
                        'total_time': data['metadata'].get('total_time', 0),
                        'condition_number': data['metadata'].get('condition_number'),
                        'method_used': data['metadata'].get('method_used', 'unknown'),
                        'n_threads': data['metadata'].get('n_threads', threads),
                        'memory_usage_mb': data['metadata'].get('memory_usage_mb', 0)
                    }
                }
        else:
            st.warning(f"Сервер вернул ошибку: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏰ Таймаут. Расчёт занимает слишком много времени.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"🔌 Не удалось подключиться к {REMOTE_SOLVER_URL}")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
        return None


def get_available_data() -> Dict:
    """Получение доступных стран и годов с сервера"""
    try:
        response = requests.get(f"{REMOTE_SOLVER_URL}/api/available", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}