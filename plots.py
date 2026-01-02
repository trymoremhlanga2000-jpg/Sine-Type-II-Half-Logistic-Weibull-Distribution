import plotly.graph_objects as go

def plot_curve(x, y, title, ylab):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title=ylab,
        template="plotly_white"
    )
    return fig




