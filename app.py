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

st.markdown("""
**Model Definition**

\\[
F(x) = \\sin\\left( \\frac{\\pi}{2}
\\frac{G(x)^\\alpha}{G(x)^\\alpha + (1-G(x))^\\alpha}
\\right),
\\quad G(x) = 1 - e^{-(x/\\lambda)^k}
\\]

This interface allows real-time sensitivity analysis of all parameters.
""")


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
# NUMERICAL MOMENTS
# =========================

mean_val = np.trapezoid(x * pdf, x)
var_val  = np.trapezoid((x - mean_val)**2 * pdf, x)
std_val  = np.sqrt(var_val)

# =========================
# METRICS DISPLAY
# =========================

st.subheader("Distribution Summary Statistics")

c1, c2, c3 = st.columns(3)

c1.metric("Mean", f"{mean_val:.4f}")
c2.metric("Variance", f"{var_val:.4f}")
c3.metric("Std. Deviation", f"{std_val:.4f}")


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
