# 🏭 Модель межотраслевого баланса Леонтьева

<div align="center">

[![Streamlit App](https://img.shields.io/badge/Streamlit-Cloud-red?style=for-the-badge&logo=streamlit)](https://leontief-model.streamlit.app)
[![Render](https://img.shields.io/badge/Render-Backend-46E3B7?style=for-the-badge&logo=render)](https://leontief-model.onrender.com/api/health)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)

**Модель "Затраты-Выпуск" Василия Леонтьева с удаленным решателем и параллельными вычислениями**

</div>

---

## 📖 О проекте

Данный проект представляет собой реализацию **модели межотраслевого баланса Леонтьева** — экономико-математической модели, позволяющей анализировать взаимосвязи между отраслями экономики.

### Основные возможности

| Функция | Описание |
|---------|----------|
| **Дашборд** | Ключевые метрики, топ-10 отраслей, распределение мультипликаторов |
| **Тепловые карты** | Визуализация матриц A и L (с логарифмической шкалой) |
| **Мультипликаторы** | Анализ мультипликаторов выпуска и затрат |
| **Сценарии** | Моделирование экономических шоков (рост/спад отраслей) |
| **Сетевой анализ** | Выявление ключевых поставщиков и потребителей |
| **Производительность** | Настройка потоков, точности, бенчмарк |
| **Системный мониторинг** | CPU, RAM, время расчётов |

### Источники данных

| Источник | Отраслей | Годы | Стран |
|----------|----------|------|-------|
| **Eurostat** | 64 | 2010-2022 | 27 стран ЕС |
| **EXIOBASE** | 200+ | 2015-2022 | 11 стран |

---

## 🚀 Демо

| Компонент | URL |
|-----------|-----|
| **Веб-интерфейс** | [https://leontief-model.streamlit.app](https://leontief-model.streamlit.app) |
| **API (здоровье)** | [https://leontief-model.onrender.com/api/health](https://leontief-model.onrender.com/api/health) |
| **GitHub репозиторий** | [https://github.com/dsdaria/leontief-model](https://github.com/dsdaria/leontief-model) |

---
## 🏗 Архитектура системы

```mermaid
flowchart TB
    subgraph Frontend ["☁️ Фронтенд (Streamlit Cloud)"]
        UI["🌐 Веб-интерфейс"]
        Client["📡 Remote Client"]
    end

    subgraph Backend ["⚙️ Бэкенд (Render.com)"]
        Flask["🚀 Flask Server"]
        Cache[("🗄️ Кэш")]
        Solver["🧮 Leontief Solver"]
        Parallel["⚡ Parallel Computing<br/>(8 threads)"]
    end

    subgraph Data ["📊 Источники данных"]
        Eurostat[("🇪🇺 Eurostat API")]
        Exiobase[("🌍 EXIOBASE API")]
    end

    User["👤 Пользователь"] --> UI
    UI --> Client
    Client -->|"POST /api/compute"| Flask
    Flask --> Cache
    Cache -->|"Данные есть"| Flask
    Cache -->|"Данных нет"| Solver
    Solver --> Parallel
    Parallel --> Eurostat
    Parallel --> Exiobase
    Solver --> Flask
    Flask --> Client
    Client --> UI
```
