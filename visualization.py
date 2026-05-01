# visualization.py
"""
Визуализация результатов
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from config import PLOTS_DIR

plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_matrix_heatmap(matrix, title, industries, filename, cmap='Blues'):
    """Тепловая карта матрицы"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Показываем не более 40 отраслей
    n = min(len(industries), 40)
    matrix_display = matrix[:n, :n]
    labels = [ind[:30] for ind in industries[:n]]
    
    im = ax.imshow(matrix_display, cmap=cmap, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    
    plt.colorbar(im, ax=ax, label='Коэффициент')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Отрасль-потребитель', fontsize=11)
    ax.set_ylabel('Отрасль-производитель', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()

def plot_multipliers(multipliers_df, filename='multipliers.png'):
    """График мультипликаторов"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Топ-10 по мультипликатору выпуска
    top_out = multipliers_df.nlargest(10, 'Мультипликатор_выпуска')
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_out)))
    ax1.barh(top_out['Отрасль'], top_out['Мультипликатор_выпуска'], color=colors)
    ax1.set_xlabel('Мультипликатор выпуска')
    ax1.set_title('🔝 Топ-10: Эффект на экономику', fontsize=12, fontweight='bold')
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    # Топ-10 по мультипликатору затрат
    top_in = multipliers_df.nlargest(10, 'Мультипликатор_затрат')
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_in)))
    ax2.barh(top_in['Отрасль'], top_in['Мультипликатор_затрат'], color=colors)
    ax2.set_xlabel('Мультипликатор затрат')
    ax2.set_title('🔝 Топ-10: Зависимость от поставок', fontsize=12, fontweight='bold')
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    plt.suptitle('📊 Отраслевые мультипликаторы', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()

def plot_shock_effects(effects, industries, filename='shock.png'):
    """График эффектов от шока"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Топ-10 затронутых отраслей
    indices = np.argsort(np.abs(effects))[-10:]
    top_industries = [industries[i] for i in indices]
    top_effects = effects[indices]
    
    colors = ['red' if x < 0 else 'green' for x in top_effects]
    bars = ax.barh(top_industries, top_effects, color=colors)
    
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Изменение выпуска (ΔX)', fontsize=11)
    ax.set_title('📈 Влияние изменения спроса на отрасли', fontsize=14, fontweight='bold')
    
    # Добавляем значения
    for bar, val in zip(bars, top_effects):
        ax.text(val, bar.get_y() + bar.get_height()/2, 
               f'{val:.1f}', ha='left' if val > 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()

def plot_condition_number(cond, filename='condition.png'):
    """График числа обусловленности"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    if cond < 1e4:
        color = 'green'
        status = '✅ Хорошо обусловлена'
    elif cond < 1e8:
        color = 'orange'
        status = '⚠️ Удовлетворительно'
    else:
        color = 'red'
        status = '❌ Плохо обусловлена'
    
    ax.barh(['Матрица (I-A)'], [cond], color=color)
    ax.set_xscale('log')
    ax.set_xlabel('Число обусловленности (логарифмическая шкала)')
    ax.set_title(f'Число обусловленности: {cond:.2e} — {status}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150)
    plt.close()

def create_all_visualizations(model, multipliers_df, shock_results=None):
    """Создание всех визуализаций"""
    matrices = model.get_matrices()
    industries = model.industries
    
    print("\n📊 Создание визуализаций...")
    
    # Тепловая карта матрицы A
    plot_matrix_heatmap(
        matrices['A'], 
        'Матрица прямых затрат A', 
        industries, 
        'matrix_A.png',
        cmap='Blues'
    )
    
    # Тепловая карта матрицы L (логарифм)
    if matrices['L'] is not None:
        plot_matrix_heatmap(
            np.log1p(matrices['L']),
            'Матрица полных затрат L (логарифм)', 
            industries, 
            'matrix_L.png',
            cmap='RdYlBu'
        )
    
    # Мультипликаторы
    plot_multipliers(multipliers_df)
    
    # Число обусловленности
    plot_condition_number(matrices['cond'])
    
    # Эффекты шоков
    if shock_results:
        for shock in shock_results:
            plot_shock_effects(shock['delta_X'], industries, f'shock_{shock["name"]}.png')
    
    print(f"✅ Все графики сохранены в {PLOTS_DIR}/")