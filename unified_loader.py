"""
Унифицированный загрузчик - поддерживает Eurostat (64) и EXIOBASE (200)
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Literal
import logging

from data_loader import EurostatDataLoader
from exiobase_loader import EXIOBASELoader
from config import MAX_INDUSTRIES, EXIOBASE_MAX_INDUSTRIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_with_source(
    country_code: str, 
    year: int, 
    source: Literal['eurostat', 'exiobase'] = 'eurostat'
) -> Dict:
    """
    Загрузка данных из выбранного источника
    
    Args:
        country_code: код страны
        year: год
        source: 'eurostat' (64 отрасли) или 'exiobase' (200 отраслей)
    
    Returns:
        Dict с матрицами Z, X, Y и списком отраслей
    """
    if source == 'exiobase':
        logger.info(f"🌍 Загрузка EXIOBASE: {country_code} {year}")
        loader = EXIOBASELoader(country_code, year)
        data = loader.get_input_output_tables()
        return data
    else:
        logger.info(f"🇪🇺 Загрузка Eurostat: {country_code} {year}")
        loader = EurostatDataLoader(country_code, year)
        data = loader.get_input_output_tables()
        # Ограничиваем 64 отраслями для Eurostat
        if data['n'] > MAX_INDUSTRIES:
            data['Z'] = data['Z'][:MAX_INDUSTRIES, :MAX_INDUSTRIES]
            data['X'] = data['X'][:MAX_INDUSTRIES]
            data['Y'] = data['Y'][:MAX_INDUSTRIES]
            data['industries'] = data['industries'][:MAX_INDUSTRIES]
            data['n'] = MAX_INDUSTRIES
        return data


def get_available_countries_for_source(source: str) -> dict:
    """Возвращает список стран для источника"""
    if source == 'exiobase':
        from config import EXIOBASE_COUNTRIES
        return EXIOBASE_COUNTRIES
    else:
        from config import AVAILABLE_COUNTRIES
        return AVAILABLE_COUNTRIES


def get_available_years_for_source(source: str) -> list:
    """Возвращает список лет для источника"""
    if source == 'exiobase':
        from config import EXIOBASE_YEARS
        return EXIOBASE_YEARS
    else:
        from config import AVAILABLE_YEARS
        return AVAILABLE_YEARS


def get_source_info(source: str) -> dict:
    """Возвращает информацию об источнике"""
    if source == 'exiobase':
        return {
            'name': '🌍 EXIOBASE',
            'description': 'Глобальная экологически-ориентированная ММВ-таблица',
            'industries': '200+',
            'years': '2015-2022',
            'countries': '11 стран (FR, DE, IT, ES, NL, PL, GB, US, CN, JP, IN)',
            'icon': '🌍'
        }
    else:
        return {
            'name': '🇪🇺 Eurostat',
            'description': 'Официальные данные Евросоюза (NACE Rev.2)',
            'industries': '64',
            'years': '2010-2022',
            'countries': '27 стран ЕС',
            'icon': '🇪🇺'
        }