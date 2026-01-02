import plotly.graph_objects as go
import numpy as np

def plot_curve(x, y, title, ylab, line_color="#0b3c5d", fill_color="#1f77b4"):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color=line_color, width=4),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate='%{x:.2f}<br>%{y:.4f}',
        name=ylab
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24, color="#0b3c5d")
        ),
        xaxis=dict(title="x", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(title=ylab, showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", size=14, color="#1f4e79"),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig
