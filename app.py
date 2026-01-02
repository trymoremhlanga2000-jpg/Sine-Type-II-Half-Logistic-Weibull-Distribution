import streamlit as st
import numpy as np

from distributions import (
    weibull_pdf, weibull_cdf, weibull_sf, weibull_hazard,
    stiiHLW_pdf, stiiHLW_cdf, stiiHLW_sf, stiiHLW_hazard
)
from plots import plot_curve

st.set_page_config(
    page_title="STIIHL Weibull Explorer",
    layout="wide"
)

st.title("Sine–Type II Half-Logistic Weibull Distribution")
st.write("Interactive exploration of shape, tail behavior, and hazard dynamics")

# =========================
# SIDEBAR PARAMETERS
# =========================

st.sidebar.header("Model Parameters")

lam = st.sidebar.slider("Scale λ", 0.5, 5.0, 1.0, 0.1)
k = st.sidebar.slider("Shape k", 0.5, 5.0, 1.5, 0.1)
alpha = st.sidebar.slider("TIIHL α", 0.1, 5.0, 1.0, 0.1)

x = np.linspace(0.001, 6 * lam, 1000)

# =========================
# DISTRIBUTION SELECTION
# =========================

dist_choice = st.radio(
    "Choose Distribution",
    ["Base Weibull", "Sine–Type II Half-Logistic Weibull"]
)

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

# =========================
# PLOTS
# =========================

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(plot_curve(x, pdf, "Probability Density Function", "f(x)"), use_container_width=True)
    st.plotly_chart(plot_curve(x, sf, "Survival Function", "S(x)"), use_container_width=True)

with col2:
    st.plotly_chart(plot_curve(x, cdf, "Cumulative Distribution Function", "F(x)"), use_container_width=True)
    st.plotly_chart(plot_curve(x, hz, "Hazard Function", "h(x)"), use_container_width=True)
