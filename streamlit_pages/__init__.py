"""
Страницы Streamlit приложения
"""

from streamlit_pages.dashboard import render_dashboard
from streamlit_pages.heatmaps import render_heatmaps
from streamlit_pages.multipliers import render_multipliers
from streamlit_pages.scenarios import render_scenarios
from streamlit_pages.network import render_network_analysis
from streamlit_pages.about import render_about

__all__ = [
    'render_dashboard',
    'render_heatmaps',
    'render_multipliers',
    'render_scenarios',
    'render_network_analysis',
    'render_about'
]