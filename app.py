import streamlit as st
import numpy as np

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

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Sine–Type II Half-Logistic Weibull Distribution")
st.write("Explore PDF, CDF, Survival, and Hazard functions interactively")

# =========================
# SIDEBAR PARAMETERS
# =========================
st.sidebar.header("Model Parameters")
lam = st.sidebar.slider("Scale λ", 0.5, 5.0, 1.0, 0.1)
k = st.sidebar.slider("Shape k", 0.5, 5.0, 1.5, 0.1)
alpha = st.sidebar.slider("TIIHL α", 0.1, 5.0, 1.0, 0.1)

x = np.linspace(0.001, 6*lam, 1000)

# Distribution choice
dist_choice = st.sidebar.radio("Distribution", ["Base Weibull", "STIIHL Weibull"])

# Compute curves
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
# METRICS
# =========================
mean_val = np.trapz(x * pdf, x)
max_pdf = max(pdf)

col1, col2, col3 = st.columns(3)
col1.metric("Mean (Approx.)", f"{mean_val:.3f}")
col2.metric("Max PDF Value", f"{max_pdf:.3f}")
col3.metric("Scale λ", f"{lam:.2f}")

# =========================
# PLOTS IN TWO ROWS
# =========================
st.subheader("Distribution Curves")

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

row1_col1.plotly_chart(plot_curve(x, pdf, "Probability Density Function", "f(x)"), use_container_width=True)
row1_col2.plotly_chart(plot_curve(x, cdf, "Cumulative Distribution Function", "F(x)"), use_container_width=True)
row2_col1.plotly_chart(plot_curve(x, sf, "Survival Function", "S(x)"), use_container_width=True)
row2_col2.plotly_chart(plot_curve(x, hz, "Hazard Function", "h(x)"), use_container_width=True)
