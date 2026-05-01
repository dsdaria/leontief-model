# config.py
"""
Конфигурация модели Леонтьева
"""

import os
from pathlib import Path

# ==================== ПАРАМЕТРЫ ЕВРОСТАТ ====================
EUROSTAT_TABLE = "naio_10_cp1700"
DEFAULT_COUNTRY = "FR"
DEFAULT_YEAR = 2020

# ==================== РАЗМЕРНОСТЬ ====================
MAX_INDUSTRIES = 64

# ==================== ПУТИ ====================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
DATA_DIR = BASE_DIR / "data"

# ==================== ДРУГИЕ ПАРАМЕТРЫ ====================
PRODUCTIVITY_THRESHOLD = 0.99

AVAILABLE_COUNTRIES = {
    "DE": "Германия",
    "FR": "Франция",
    "IT": "Италия",
    "ES": "Испания",
    "NL": "Нидерланды",
    "PL": "Польша",
    "BE": "Бельгия",
    "AT": "Австрия",
    "SE": "Швеция",
    "DK": "Дания",
    "CZ": "Чехия",
    "PT": "Португалия",
    "FI": "Финляндия",
    "RO": "Румыния",
    "HU": "Венгрия",
    "IE": "Ирландия",
    "SK": "Словакия",
    "BG": "Болгария",
    "HR": "Хорватия",
    "LT": "Литва",
    "SI": "Словения",
    "LV": "Латвия",
    "EE": "Эстония",
    "CY": "Кипр",
    "LU": "Люксембург",
    "MT": "Мальта",
}

AVAILABLE_YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


def ensure_directories():
    """Создаёт необходимые директории"""
    for directory in [OUTPUT_DIR, PLOTS_DIR, DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_country_name(code):
    """Возвращает название страны по коду"""
    return AVAILABLE_COUNTRIES.get(code, code)



# ==================== ПОДДЕРЖКА EXIOBASE ====================
EXIOBASE_ENABLED = True
EXIOBASE_MAX_INDUSTRIES = 200

# Доступные страны для EXIOBASE
EXIOBASE_COUNTRIES = {
    "FR": "Франция",
    "DE": "Германия", 
    "IT": "Италия",
    "ES": "Испания",
    "NL": "Нидерланды",
    "PL": "Польша",
    "GB": "Великобритания",
    "US": "США",
    "CN": "Китай",
    "JP": "Япония",
    "IN": "Индия",
}

EXIOBASE_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]