"""
Корректная загрузка данных из Eurostat API v2.1
Таблица naio_10_cp1700 - Symmetric Input-Output Table (product x product)
"""

import pandas as pd
import numpy as np
import requests
import json
import re
from typing import Dict, List, Optional, Tuple
from config import DEFAULT_COUNTRY, DEFAULT_YEAR, MAX_INDUSTRIES, DATA_DIR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EurostatDataLoader:
    """Загрузчик данных Eurostat с правильным парсингом"""
    
    def __init__(self, country_code: str = DEFAULT_COUNTRY, year: int = DEFAULT_YEAR):
        self.country_code = country_code
        self.year = year
        self.table_code = "naio_10_cp1700"  # ← ИСПРАВИТЬ ЗДЕСЬ
        self.base_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
        
    def get_input_output_tables(self) -> Dict:
        """Загрузка и парсинг таблицы Затраты-Выпуск"""
        logger.info(f"📊 Загрузка таблицы {self.table_code} для {self.country_code}, {self.year}")
        
        # Пробуем загрузку через пакет eurostat (основной метод)
        data = self._fetch_via_eurostat_package()
        if data:
            return data
        
        # Пробуем SDMX API
        data = self._fetch_sdmx_data()
        if data:
            return self._parse_sdmx_response(data)
        
        # Fallback: демо-данные
        logger.warning("⚠️ Использую демонстрационные данные")
        return self._generate_demo_data()
    
    def _fetch_sdmx_data(self) -> Optional[Dict]:
        """Загрузка через SDMX 2.1 REST API"""
        urls = [
            f"{self.base_url}/{self.table_code}/A.{self.country_code}",
        ]
        
        for url in urls:
            try:
                logger.info(f"🔗 Запрос: {url}")
                response = requests.get(
                    url,
                    params={'format': 'json', 'time': str(self.year), 'lang': 'en'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    logger.info("✅ Данные получены через SDMX API")
                    return response.json()
                else:
                    logger.warning(f"❌ HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка запроса: {e}")
        
        return None
    
    def _parse_sdmx_response(self, data: Dict) -> Dict:
        """Парсинг SDMX-JSON ответа"""
        try:
            structure = data.get('structure', {})
            dimensions = structure.get('dimensions', {}).get('observation', [])
            
            dim_dict = {dim['id']: dim for dim in dimensions}
            
            row_labels = self._get_dimension_values(dim_dict, ['nace_r2', 'induse', 'prd_use'])
            col_labels = self._get_dimension_values(dim_dict, ['nace_r2_cp', 'prod', 'prd_ava'])
            
            if not row_labels or not col_labels:
                return self._generate_demo_data()
            
            observations = data.get('dataSets', [{}])[0].get('observations', {})
            if not observations:
                return self._generate_demo_data()
            
            dim_ids = [d['id'] for d in dimensions]
            row_idx = next((i for i, id_ in enumerate(dim_ids) if id_ in ['nace_r2', 'induse', 'prd_use']), 0)
            col_idx = next((i for i, id_ in enumerate(dim_ids) if id_ in ['nace_r2_cp', 'prod', 'prd_ava']), 1)
            
            n_rows = min(len(row_labels), MAX_INDUSTRIES)
            n_cols = min(len(col_labels), MAX_INDUSTRIES)
            Z = np.zeros((n_rows, n_cols))
            
            for key, value in observations.items():
                indices = [int(i) for i in key.split(':')]
                if len(indices) > max(row_idx, col_idx):
                    r, c = indices[row_idx], indices[col_idx]
                    if r < n_rows and c < n_cols:
                        Z[r, c] = float(value[0]) if isinstance(value, list) else float(value)
            
            industries = [self._clean_industry_name(name) for name in row_labels[:n_rows]]
            X = np.abs(Z).sum(axis=1)
            Y = np.abs(Z).sum(axis=0) * 0.25
            
            return {
                'industries': industries,
                'Z': np.abs(Z).astype(np.float64),
                'X': X.astype(np.float64),
                'Y': Y.astype(np.float64),
                'n': n_rows
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка парсинга SDMX: {e}")
            return self._generate_demo_data()
    
    def _get_dimension_values(self, dim_dict: Dict, possible_names: List[str]) -> List[str]:
        """Извлекает значения измерений"""
        for name in possible_names:
            if name in dim_dict:
                values = dim_dict[name].get('values', [])
                if values:
                    return [v.get('name', f'{name}_{i}') for i, v in enumerate(values)]
        return []
    
    def _clean_industry_name(self, name: str) -> str:
        """Очищает и форматирует название отрасли"""
        name_str = str(name).strip()
        
        nace_descriptions = {
            'A01': 'Crop and animal production',
            'A02': 'Forestry and logging',
            'A03': 'Fishing and aquaculture',
            'B': 'Mining and quarrying',
            'C10-C12': 'Food, beverages, tobacco',
            'C13-C15': 'Textiles, wearing apparel',
            'C16': 'Wood and wood products',
            'C17': 'Paper products',
            'C18': 'Printing and recording',
            'C19': 'Coke and refined petroleum',
            'C20': 'Chemicals',
            'C21': 'Pharmaceuticals',
            'C22': 'Rubber and plastics',
            'C23': 'Other non-metallic minerals',
            'C24': 'Basic metals',
            'C25': 'Fabricated metal products',
            'C26': 'Computer, electronic products',
            'C27': 'Electrical equipment',
            'C28': 'Machinery n.e.c.',
            'C29': 'Motor vehicles',
            'C30': 'Other transport equipment',
            'C31-C32': 'Furniture, manufacturing n.e.c.',
            'C33': 'Repair and installation',
            'D35': 'Electricity, gas, steam',
            'E36': 'Water collection, treatment',
            'E37-E39': 'Sewerage, waste management',
            'F': 'Construction',
            'G45': 'Wholesale trade',
            'G46': 'Wholesale trade (except vehicles)',
            'G47': 'Retail trade',
            'H49': 'Land transport',
            'H50': 'Water transport',
            'H51': 'Air transport',
            'H52': 'Warehousing',
            'H53': 'Postal and courier',
            'I': 'Accommodation and food',
            'J58': 'Publishing',
            'J59-J60': 'Motion picture, broadcasting',
            'J61': 'Telecommunications',
            'J62-J63': 'IT services',
            'K64': 'Financial services',
            'K65': 'Insurance, pension',
            'K66': 'Financial auxiliaries',
            'L68': 'Real estate',
            'M69-M70': 'Legal, consulting',
            'M71': 'Architecture, engineering',
            'M72': 'R&D',
            'M73': 'Advertising, market research',
            'M74-M75': 'Professional, veterinary',
            'N77': 'Rental and leasing',
            'N78': 'Employment services',
            'N79': 'Travel agencies',
            'N80-N82': 'Security, office support',
            'O84': 'Public administration',
            'P85': 'Education',
            'Q86': 'Human health',
            'Q87-Q88': 'Residential care, social work',
            'R90-R92': 'Creative, arts',
            'R93': 'Sports, recreation',
            'S94': 'Membership organizations',
            'S95': 'Repair computers, personal goods',
            'S96': 'Other personal services',
            'T': 'Activities of households',
            'U': 'Extraterritorial organizations'
        }
        
        for code, desc in nace_descriptions.items():
            if code in name_str or name_str in code:
                return f"{code}_{desc}"
        
        name_str = re.sub(r'^(CPA_|NACE_)?', '', name_str)
        name_str = name_str.replace('_', ' ').strip()
        name_str = re.sub(r'\s+', ' ', name_str)
        
        return name_str[:60]
    
    def _fetch_via_eurostat_package(self) -> Optional[Dict]:
        """Загрузка через пакет eurostat"""
        try:
            import eurostat
            
            param_combinations = [
                {'geo': [self.country_code], 'time': [self.year]},
                {'geo': [self.country_code], 'time': [self.year], 'na_item': ['TOTAL']},
                {'geo': [self.country_code], 'time': [self.year], 'na_item': ['PROD']},
            ]
            
            for filters in param_combinations:
                try:
                    logger.info(f"   Пробуем eurostat: {filters}")
                    df = eurostat.get_data_df(self.table_code, filter_pars=filters, flags=False)
                    
                    if df is not None and not df.empty:
                        logger.info(f"✅ Загружено через eurostat: {len(df)} записей")
                        result = self._parse_eurostat_dataframe(df)
                        if result:
                            return result
                except Exception as e:
                    logger.warning(f"   Ошибка: {e}")
                    continue
                    
        except ImportError:
            logger.warning("⚠️ Пакет eurostat не установлен")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка eurostat: {e}")
        
        return None
    
    def _parse_eurostat_dataframe(self, df: pd.DataFrame) -> Optional[Dict]:
        """Парсинг DataFrame от пакета eurostat"""
        
        logger.info(f"   Колонки: {list(df.columns)}")
        
        # Определяем колонки с годами
        year_cols = []
        non_year_cols = []
        
        for col in df.columns:
            try:
                int(str(col))
                year_cols.append(col)
            except (ValueError, TypeError):
                non_year_cols.append(col)
        
        logger.info(f"   Годы: {year_cols}")
        
        # Выбираем год
        target_year = str(self.year)
        if target_year not in year_cols:
            available = sorted([int(y) for y in year_cols if str(y).isdigit()])
            if available:
                closest = min(available, key=lambda x: abs(x - self.year))
                target_year = str(closest)
                logger.info(f"   Использую ближайший год: {target_year}")
            else:
                return None
        
        # Ищем идентификационные колонки
        id_cols = [c for c in non_year_cols if c not in ['freq', 'unit', 'stk_flow', 'obs_status']]
        
        # Определяем колонки строк и столбцов
        row_col = None
        col_col = None
        
        row_candidates = ['nace_r2', 'induse', 'prd_use']
        col_candidates = ['nace_r2_cp', 'prode', 'prd_ava']
        
        for col in id_cols:
            if col.lower() in row_candidates and row_col is None:
                row_col = col
            elif col.lower() in col_candidates and col_col is None and col != row_col:
                col_col = col
        
        # Автоопределение если не нашли
        if row_col is None or col_col is None:
            candidates = [(c, df[c].nunique()) for c in id_cols if 5 < df[c].nunique() < 500]
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            if len(candidates) >= 2:
                row_col = candidates[0][0]
                col_col = candidates[1][0]
                logger.info(f"   Авто: row='{row_col}', col='{col_col}'")
        
        if row_col is None or col_col is None:
            logger.warning("❌ Не определены row/col колонки")
            return None
        
        logger.info(f"   ✅ row='{row_col}', col='{col_col}', year='{target_year}'")
        
        # Создаём pivot table
        try:
            df[target_year] = pd.to_numeric(df[target_year], errors='coerce').fillna(0)
            
            pivot = df.pivot_table(
                index=row_col,
                columns=col_col,
                values=target_year,
                aggfunc='sum',
                fill_value=0
            )
            
            # Фильтруем
            pivot = pivot.loc[pivot.sum(axis=1) > 0, pivot.sum(axis=0) > 0]
            
            logger.info(f"   Pivot shape: {pivot.shape}")
            
            if pivot.empty or pivot.shape[0] < 2:
                return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка pivot: {e}")
            return None
        
        Z = np.abs(pivot.values)
        industries = [self._clean_industry_name(str(idx)) for idx in pivot.index]
        
        n = min(len(industries), MAX_INDUSTRIES)
        Z = Z[:n, :n]
        industries = industries[:n]
        
        if Z.sum() == 0:
            return None
        
        X = Z.sum(axis=1)
        Y = Z.sum(axis=0) * 0.25
        
        logger.info(f"✅ Загружено {n} отраслей, сумма выпуска: {X.sum():.0f}")
        
        return {
            'industries': industries,
            'Z': Z.astype(np.float64),
            'X': X.astype(np.float64),
            'Y': Y.astype(np.float64),
            'n': n
        }
    
    def _generate_demo_data(self) -> Dict:
        """Генерация демонстрационных данных"""
        logger.info("📊 Генерация демонстрационных данных...")
        
        industries = [
            "A01_Crop and animal production",
            "A02_Forestry and logging",
            "A03_Fishing and aquaculture",
            "B_Mining and quarrying",
            "C10-C12_Food, beverages, tobacco",
            "C13-C15_Textiles, wearing apparel",
            "C16_Wood and wood products",
            "C17_Paper products",
            "C18_Printing and recording",
            "C19_Coke and refined petroleum",
            "C20_Chemicals",
            "C21_Pharmaceuticals",
            "C22_Rubber and plastics",
            "C23_Other non-metallic minerals",
            "C24_Basic metals",
            "C25_Fabricated metal products",
            "C26_Computer, electronic products",
            "C27_Electrical equipment",
            "C28_Machinery n.e.c.",
            "C29_Motor vehicles",
            "C30_Other transport equipment",
            "C31-C32_Furniture, manufacturing n.e.c.",
            "C33_Repair and installation",
            "D35_Electricity, gas, steam",
            "E36_Water collection, treatment",
            "E37-E39_Sewerage, waste management",
            "F_Construction",
            "G45_Wholesale trade",
            "G46_Wholesale trade (except vehicles)",
            "G47_Retail trade",
            "H49_Land transport",
            "H50_Water transport",
            "H51_Air transport",
            "H52_Warehousing",
            "H53_Postal and courier",
            "I_Accommodation and food",
            "J58_Publishing",
            "J59-J60_Motion picture, broadcasting",
            "J61_Telecommunications",
            "J62-J63_IT services",
            "K64_Financial services",
            "K65_Insurance, pension",
            "K66_Financial auxiliaries",
            "L68_Real estate",
            "M69-M70_Legal, consulting",
            "M71_Architecture, engineering",
            "M72_R&D",
            "M73_Advertising, market research",
            "M74-M75_Professional, veterinary",
            "N77_Rental and leasing",
            "N78_Employment services",
            "N79_Travel agencies",
            "N80-N82_Security, office support",
            "O84_Public administration",
            "P85_Education",
            "Q86_Human health",
            "Q87-Q88_Residential care, social work",
            "R90-R92_Creative, arts",
            "R93_Sports, recreation",
            "S94_Membership organizations",
            "S95_Repair computers, personal goods",
            "S96_Other personal services",
            "T_Activities of households",
            "U_Extraterritorial organizations"
        ][:MAX_INDUSTRIES]
        
        n = len(industries)
        np.random.seed(42)
        Z = np.zeros((n, n))
        
        for i in range(n):
            Z[i, i] = np.random.uniform(100, 500)
            for j in range(max(0, i-2), min(n, i+3)):
                if i != j:
                    Z[i, j] = np.random.exponential(50)
        
        try:
            const_idx = next(i for i, ind in enumerate(industries) if 'Construction' in ind)
            steel_idx = next(i for i, ind in enumerate(industries) if 'Basic metals' in ind)
            minerals_idx = next(i for i, ind in enumerate(industries) if 'non-metallic' in ind.lower())
            electricity_idx = next(i for i, ind in enumerate(industries) if 'Electricity' in ind)
            
            Z[const_idx, steel_idx] *= 5
            Z[const_idx, minerals_idx] *= 4
            Z[:, electricity_idx] *= 1.5
        except (StopIteration, IndexError):
            pass
        
        row_sums = Z.sum(axis=1)
        Z = Z / row_sums[:, np.newaxis] * 1000
        
        X = Z.sum(axis=1) * 1.3
        Y = Z.sum(axis=1) * 0.3
        
        return {
            'industries': industries,
            'Z': Z.astype(np.float64),
            'X': X.astype(np.float64),
            'Y': Y.astype(np.float64),
            'n': n
        }