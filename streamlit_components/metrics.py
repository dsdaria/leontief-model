import streamlit as st
from typing import List, Dict

def create_metric_card(icon: str, value: str, label: str, sublabel: str = "") -> str:
    return f"""
    <div style="
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #dee2e6;
        transition: transform 0.2s;
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <h3 style="color: #2a5298; margin: 0.5rem 0; font-size: 1.75rem;">{value}</h3>
        <div style="color: #495057; font-weight: 500;">{label}</div>
        {f'<div style="color: #6c757d; font-size: 0.8rem; margin-top: 0.25rem;">{sublabel}</div>' if sublabel else ''}
    </div>
    """

def render_metrics_row(metrics: List[Dict], columns: int = 4):
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        if i < len(cols):
            with cols[i]:
                st.markdown(
                    create_metric_card(
                        metric.get('icon', '📊'),
                        str(metric.get('value', 'N/A')),
                        metric.get('label', ''),
                        metric.get('sublabel', '')
                    ),
                    unsafe_allow_html=True
                )