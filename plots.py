import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

def plot_curve(x, y, title, y_label):
    """Create a standard plotly curve"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color='#f5c77a', width=3),
        fill='tozeroy',
        fillcolor='rgba(245, 199, 122, 0.2)',
        name=y_label
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title=y_label,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f5c77a', size=12),
        xaxis=dict(
            gridcolor='rgba(245, 199, 122, 0.1)',
            linecolor='rgba(245, 199, 122, 0.3)',
            zerolinecolor='rgba(245, 199, 122, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(245, 199, 122, 0.1)',
            linecolor='rgba(245, 199, 122, 0.3)',
            zerolinecolor='rgba(245, 199, 122, 0.2)'
        ),
        hovermode='x unified',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def plot_comparison(x, y1, y2, label1, label2, title):
    """Plot comparison between two curves"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y1,
        mode='lines',
        line=dict(color='#f5c77a', width=3),
        name=label1,
        fill='tozeroy',
        fillcolor='rgba(245, 199, 122, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y2,
        mode='lines',
        line=dict(color='#8B5A2B', width=3, dash='dash'),
        name=label2,
        fill='tozeroy',
        fillcolor='rgba(139, 90, 43, 0.2)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='Value',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f5c77a', size=12),
        xaxis=dict(
            gridcolor='rgba(245, 199, 122, 0.1)',
            linecolor='rgba(245, 199, 122, 0.3)',
            zerolinecolor='rgba(245, 199, 122, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(245, 199, 122, 0.1)',
            linecolor='rgba(245, 199, 122, 0.3)',
            zerolinecolor='rgba(245, 199, 122, 0.2)'
        ),
        legend=dict(
            font=dict(color='#f5c77a'),
            bgcolor='rgba(20, 20, 20, 0.7)',
            bordercolor='rgba(245, 199, 122, 0.3)'
        ),
        hovermode='x unified',
        height=400
    )
    
    return fig

def plot_histogram_with_fit(data, fitted_pdf, x_range, title):
    """Plot histogram with fitted PDF overlay"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data,
            name='Data Histogram',
            nbinsx=30,
            histnorm='probability density',
            marker_color='rgba(245, 199, 122, 0.6)',
            marker_line=dict(color='#f5c77a', width=1),
            opacity=0.7
        ),
        secondary_y=False
    )
    
    # Fitted PDF
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=fitted_pdf,
            mode='lines',
            line=dict(color='#8B5A2B', width=3),
            name='Fitted STIIHL Weibull',
            fill='tozeroy',
            fillcolor='rgba(139, 90, 43, 0.2)'
        ),
        secondary_y=False
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Value',
        yaxis_title='Density',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f5c77a', size=12),
        barmode='overlay',
        bargap=0.1,
        showlegend=True,
        legend=dict(
            font=dict(color='#f5c77a'),
            bgcolor='rgba(20, 20, 20, 0.7)',
            bordercolor='rgba(245, 199, 122, 0.3)'
        ),
        height=450
    )
    
    fig.update_xaxes(
        gridcolor='rgba(245, 199, 122, 0.1)',
        linecolor='rgba(245, 199, 122, 0.3)'
    )
    
    fig.update_yaxes(
        title_text='Density',
        secondary_y=False,
        gridcolor='rgba(245, 199, 122, 0.1)',
        linecolor='rgba(245, 199, 122, 0.3)'
    )
    
    return fig

def plot_qq(data, fitted_quantiles, title):
    """Plot Q-Q plot for goodness of fit"""
    fig = go.Figure()
    
    # Q-Q line
    min_val = min(min(data), min(fitted_quantiles))
    max_val = max(max(data), max(fitted_quantiles))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='#ef4444', width=2, dash='dash'),
        name='Reference Line (y=x)'
    ))
    
    # Q-Q points
    fig.add_trace(go.Scatter(
        x=sorted(data),
        y=sorted(fitted_quantiles),
        mode='markers',
        marker=dict(
            color='#f5c77a',
            size=8,
            line=dict(color='#d4a94e', width=1)
        ),
        name='Q-Q Points'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Empirical Quantiles',
        yaxis_title='Theoretical Quantiles (STIIHL Weibull)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f5c77a', size=12),
        xaxis=dict(
            gridcolor='rgba(245, 199, 122, 0.1)',
            linecolor='rgba(245, 199, 122, 0.3)'
        ),
        yaxis=dict(
            gridcolor='rgba(245, 199, 122, 0.1)',
            linecolor='rgba(245, 199, 122, 0.3)'
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='#f5c77a'),
            bgcolor='rgba(20, 20, 20, 0.7)',
            bordercolor='rgba(245, 199, 122, 0.3)'
        ),
        height=450
    )
    
    return fig