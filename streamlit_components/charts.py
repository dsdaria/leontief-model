import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_heatmap_plotly(
    matrix: pd.DataFrame, 
    title: str, 
    colorscale: str = 'Viridis',
    n_show: int = 30,
    use_log: bool = True
) -> go.Figure:
    """Создание тепловой карты с динамическим размером"""
    
    n = min(n_show, len(matrix))
    df = matrix.iloc[:n, :n].copy()
    
    if use_log:
        values = np.log1p(df.values)
        colorbar_title = "log(1 + коэф.)"
    else:
        values = df.values
        colorbar_title = "Коэффициент"
    
    # Увеличиваем высоту для большего количества отраслей
    height = max(500, n * 12)
    
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=df.columns,
        y=df.index,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
        hovertemplate=(
            "<b>Производитель:</b> %{y}<br>" +
            "<b>Потребитель:</b> %{x}<br>" +
            "<b>Значение:</b> %{z:.4f}<br>" +
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=height,
        template='plotly_white',
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), title="Отрасль-потребитель"),
        yaxis=dict(tickfont=dict(size=9), title="Отрасль-производитель"),
        margin=dict(l=100, r=50, t=80, b=100)
    )
    
    return fig

def create_bar_chart_horizontal(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    top_n: int = 10,
    color_scale: str = 'Viridis',
    height: int = 500
) -> go.Figure:
    df_top = data.nlargest(top_n, x_col)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_top[x_col],
            y=df_top[y_col],
            orientation='h',
            marker=dict(
                color=df_top[x_col],
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(title=x_col)
            ),
            text=df_top[x_col].round(3),
            textposition='outside',
            hovertemplate=(
                f"<b>Отрасль:</b> %{{y}}<br>" +
                f"<b>{x_col}:</b> %{{x:.3f}}<br>" +
                "<extra></extra>"
            )
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=height,
        template='plotly_white',
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(l=200, r=50, t=50, b=50)
    )
    
    return fig

def create_scenario_analysis(
    df: pd.DataFrame,
    scenario_name: str,
    top_n: int = 20
) -> go.Figure:
    # Проверяем название колонки с изменениями
    value_col = 'Изменение_выпуска' if 'Изменение_выпуска' in df.columns else df.columns[-1]
    industry_col = 'Отрасль' if 'Отрасль' in df.columns else df.columns[0]
    
    df_sorted = df.sort_values(value_col, ascending=True)
    top_negative = df_sorted.head(top_n // 2)
    top_positive = df_sorted.tail(top_n // 2)
    df_plot = pd.concat([top_negative, top_positive])
    
    colors = ['#dc3545' if x < 0 else '#28a745' for x in df_plot[value_col]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_plot[value_col],
            y=df_plot[industry_col],
            orientation='h',
            marker_color=colors,
            text=df_plot[value_col].round(1),
            textposition='outside',
            hovertemplate=(
                "<b>Отрасль:</b> %{y}<br>" +
                "<b>Изменение:</b> %{x:,.1f} млн €<br>" +
                "<extra></extra>"
            )
        )
    ])
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=dict(text=f"Влияние сценария: {scenario_name}", font=dict(size=16)),
        height=600,
        template='plotly_white',
        xaxis_title="Изменение выпуска (ΔX)",
        yaxis_title="Отрасль",
        margin=dict(l=200, r=50, t=50, b=50),
        showlegend=False
    )
    
    return fig

def create_sankey_diagram(
    matrix: pd.DataFrame,
    top_n: int = 15,
    threshold_percentile: int = 90
) -> go.Figure:
    sums = matrix.sum()
    top_industries = sums.nlargest(top_n).index
    
    threshold = np.percentile(matrix.values.flatten(), threshold_percentile)
    
    sources = []
    targets = []
    values = []
    
    for i, source in enumerate(top_industries):
        for j, target in enumerate(top_industries):
            val = matrix.loc[source, target]
            if val > threshold and val > 0:
                sources.append(i)
                targets.append(j)
                values.append(val)
    
    if not sources:
        return None
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[s[:30] for s in top_industries],
            color='lightblue'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(74, 144, 226, 0.4)'
        )
    )])
    
    fig.update_layout(
        title=f"Ключевые межотраслевые связи (выше {threshold_percentile} перцентиля)",
        font=dict(size=10),
        height=700,
        margin=dict(l=200, r=50, t=50, b=50)
    )
    
    return fig