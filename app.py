import streamlit as st
import numpy as np
import plotly.graph_objects as go

from distributions import (
    weibull_pdf, weibull_cdf, weibull_sf, weibull_hazard,
    stiiHLW_pdf, stiiHLW_cdf, stiiHLW_sf, stiiHLW_hazard
)
from plots import plot_curve

st.set_page_config(
    page_title="STIIHL Weibull Explorer",
    page_icon="📊",
    layout="wide"
)

# ===== Sidebar =====
st.sidebar.title("STIIHL Weibull Explorer")
st.sidebar.markdown("""
Explore PDF, CDF, Survival, and Hazard functions interactively.
Adjust the parameters below and observe how the distribution changes.
""")

# Model parameters
lam = st.sidebar.slider("Scale λ", 0.5, 5.0, 1.0, 0.1)
k = st.sidebar.slider("Shape k", 0.5, 5.0, 1.5, 0.1)
alpha = st.sidebar.slider("TIIHL α", 0.1, 5.0, 1.0, 0.1)

dist_choice = st.sidebar.radio("Distribution", ["Base Weibull", "STIIHL Weibull"])

# ===== Main Title & Formula =====
st.title("Sine–Type II Half-Logistic Weibull Distribution")

st.subheader("Model Definition")
st.latex(r"""
F(x) = \sin\left( \frac{\pi}{2} 
\frac{G(x)^\alpha}{G(x)^\alpha + (1-G(x))^\alpha} \right), \quad
G(x) = 1 - e^{-(x/\lambda)^k}
""")
st.write("This interface allows real-time sensitivity analysis of all parameters.")

# ===== Compute distribution =====
x = np.linspace(0.001, 6*lam, 1000)

if dist_choice == "Base Weibull":
    pdf = weibull_pdf(x, lam, k)
    cdf = weibull_cdf(x, lam, k)
    sf = weibull_sf(x, lam, k)
    hz = weibull_hazard(x, lam, k)
else:
    pdf = stiiHLW_pdf(x, lam, k, alpha)
    cdf = stiiHLW_cdf(x, lam, k, alpha)
    sf = stiiHLW_sf(x, lam, k, alpha)
    hz = stiiHLW_hazard(x, lam, k, alpha)

# ===== Summary Metrics =====
mean_val = np.trapz(x * pdf, x)
var_val  = np.trapz((x - mean_val)**2 * pdf, x)
std_val  = np.sqrt(var_val)

st.subheader("Distribution Summary Statistics")
col1, col2, col3 = st.columns(3)

col1.metric("Mean", f"{mean_val:.4f}")
col2.metric("Variance", f"{var_val:.4f}")
col3.metric("Std Deviation", f"{std_val:.4f}")

# ===== Tabs for Plots =====
st.subheader("Distribution Curves")
tab_pdf, tab_cdf, tab_sf, tab_hz = st.tabs([
    "Probability Density Function",
    "Cumulative Distribution Function",
    "Survival Function",
    "Hazard Function"
])

tab_pdf.plotly_chart(plot_curve(x, pdf, "Probability Density Function", "f(x)"), use_container_width=True)
tab_cdf.plotly_chart(plot_curve(x, cdf, "Cumulative Distribution Function", "F(x)"), use_container_width=True)
tab_sf.plotly_chart(plot_curve(x, sf, "Survival Function", "S(x)"), use_container_width=True)
tab_hz.plotly_chart(plot_curve(x, hz, "Hazard Function", "h(x)"), use_container_width=True)
