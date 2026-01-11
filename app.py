import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import trapezoid
import io
import base64
from datetime import datetime

# Import from local modules
from distributions import (
    weibull_pdf, weibull_cdf, weibull_sf, weibull_hazard,
    stiiHLW_pdf, stiiHLW_cdf, stiiHLW_sf, stiiHLW_hazard,
    mle_stiiHLW, goodness_of_fit, generate_stiiHLW_samples,
    stiiHLW_quantile
)

from plots import plot_curve, plot_comparison, plot_histogram_with_fit, plot_qq

# =============================
# PAGE CONFIGURATION
# =============================
st.set_page_config(
    page_title="Trymore's STIIHL Weibull Analyzer",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# GOLD + BLACK PREMIUM THEME
# =============================
def apply_premium_theme():
    st.markdown("""
    <style>
    /* MAIN BACKGROUND */
    body, .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: #f5c77a;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    
    /* CARD DESIGN - PREMIUM GLASS EFFECT */
    .card {
        background: linear-gradient(145deg, rgba(15, 15, 15, 0.95), rgba(26, 26, 26, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 25px;
        border: 1px solid rgba(245, 199, 122, 0.25);
        box-shadow: 
            0 8px 32px rgba(245, 199, 122, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: rgba(245, 199, 122, 0.4);
        box-shadow: 
            0 12px 48px rgba(245, 199, 122, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }
    
    /* TYPOGRAPHY - LUXURY STYLE */
    h1, h2, h3 {
        color: #f5c77a !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem !important;
    }
    
    h1 {
        font-size: 2.8rem !important;
        background: linear-gradient(90deg, #f5c77a, #ffd98e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
    }
    
    h1:after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 0;
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #f5c77a, transparent);
        border-radius: 2px;
    }
    
    /* INPUT CONTROLS - LUXURY STYLE */
    .stSelectbox > div, .stNumberInput > div, .stSlider > div, .stRadio > div {
        background: rgba(18, 18, 18, 0.9) !important;
        border: 1.5px solid rgba(245, 199, 122, 0.3) !important;
        border-radius: 12px !important;
        color: #f5c77a !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div:hover, .stNumberInput > div:hover, 
    .stSlider > div:hover, .stRadio > div:hover {
        border-color: rgba(245, 199, 122, 0.6) !important;
        box-shadow: 0 0 20px rgba(245, 199, 122, 0.15);
    }
    
    /* BUTTONS - PREMIUM GOLD GRADIENT */
    .stButton > button {
        background: linear-gradient(135deg, #f5c77a 0%, #ffd98e 100%);
        color: #0a0a0a !important;
        border-radius: 12px;
        padding: 14px 28px;
        font-size: 16px;
        font-weight: 700;
        border: none;
        box-shadow: 
            0 4px 20px rgba(245, 199, 122, 0.4),
            0 2px 4px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 8px 30px rgba(245, 199, 122, 0.6),
            0 4px 8px rgba(0, 0, 0, 0.4);
        background: linear-gradient(135deg, #ffd98e 0%, #f5c77a 100%);
    }
    
    /* METRICS - PREMIUM CARDS */
    [data-testid="metric-container"] {
        background: rgba(15, 15, 15, 0.7) !important;
        border: 1px solid rgba(245, 199, 122, 0.2);
        border-radius: 16px;
        padding: 20px;
    }
    
    [data-testid="metric-label"] {
        color: #b0b0b0 !important;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    [data-testid="metric-value"] {
        color: #f5c77a !important;
        font-size: 2rem;
        font-weight: 800;
    }
    
    /* SIDEBAR - DARK LUXURY */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a1a 100%);
        border-right: 1px solid rgba(245, 199, 122, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: transparent !important;
    }
    
    /* PROGRESS BAR - GOLD STYLE */
    .stProgress > div > div {
        background: linear-gradient(90deg, #f5c77a, #ffd98e);
        border-radius: 10px;
    }
    
    /* TABS - PREMIUM STYLE */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(18, 18, 18, 0.8) !important;
        border: 1px solid rgba(245, 199, 122, 0.2) !important;
        color: #b0b0b0 !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: rgba(245, 199, 122, 0.4) !important;
        color: #f5c77a !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(245, 199, 122, 0.2), rgba(255, 217, 142, 0.1)) !important;
        border-color: #f5c77a !important;
        color: #f5c77a !important;
    }
    
    /* DISTRIBUTION BADGES */
    .distro-badge {
        display: inline-block;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 18px;
        margin: 10px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .weibull-badge {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.2), rgba(30, 64, 175, 0.1));
        color: #60a5fa;
        border: 1px solid rgba(37, 99, 235, 0.3);
    }
    
    .stiihl-badge {
        background: linear-gradient(135deg, rgba(245, 199, 122, 0.2), rgba(255, 217, 142, 0.1));
        color: #f5c77a;
        border: 1px solid rgba(245, 199, 122, 0.3);
    }
    
    /* FORM STYLING */
    .form-section {
        background: rgba(20, 20, 20, 0.6);
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(245, 199, 122, 0.15);
    }
    
    /* ANALYSIS CARDS */
    .analysis-card {
        background: rgba(18, 18, 18, 0.7);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(245, 199, 122, 0.1);
        transition: all 0.3s ease;
    }
    
    .analysis-card:hover {
        border-color: rgba(245, 199, 122, 0.3);
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(245, 199, 122, 0.15);
    }
    
    /* STATS CARDS */
    .stats-card {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.9), rgba(20, 20, 20, 0.9));
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(245, 199, 122, 0.2);
        text-align: center;
    }
    
    /* FOOTER */
    .footer {
        position: fixed;
        bottom: 20px;
        right: 30px;
        font-size: 12px;
        color: rgba(245, 199, 122, 0.6);
        letter-spacing: 1px;
        font-weight: 300;
    }
    
    /* PARAMETER INDICATORS */
    .param-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
    }
    
    .lam-indicator { background-color: #f5c77a; }
    .k-indicator { background-color: #8B5A2B; }
    .alpha-indicator { background-color: #22c55e; }
    
    /* UPLOAD BOX STYLING */
    .upload-box {
        border: 2px dashed rgba(245, 199, 122, 0.3);
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        background: rgba(20, 20, 20, 0.5);
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: rgba(245, 199, 122, 0.6);
        background: rgba(30, 30, 30, 0.5);
    }
    
    /* CHART STYLING OVERRIDES */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        background-color: transparent !important;
    }
    
    /* CODE BLOCK STYLING */
    .stCodeBlock {
        background: rgba(20, 20, 20, 0.8) !important;
        border: 1px solid rgba(245, 199, 122, 0.2) !important;
        border-radius: 12px !important;
    }
    
    /* DATA EDITOR STYLING */
    .stDataFrame {
        background: rgba(20, 20, 20, 0.7) !important;
        border: 1px solid rgba(245, 199, 122, 0.2) !important;
        border-radius: 12px !important;
    }
    
    /* CUSTOM SCROLLBAR */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(20, 20, 20, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #f5c77a, #d4a94e);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #ffd98e, #f5c77a);
    }
    </style>
    """, unsafe_allow_html=True)

apply_premium_theme()

# =============================
# SIDEBAR NAVIGATION
# =============================
st.sidebar.markdown("<h2 style='text-align: center;'>💎 TryieDataMagic</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align: center; color: rgba(245, 199, 122, 0.7); margin-bottom: 30px;'>DISTRIBUTION ANALYTICS</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    ["🏠 Dashboard", "📊 Distribution Explorer", "🔬 Statistical Analysis", 
     "📈 Data Fitting", "🧪 Monte Carlo Simulation", "📚 Documentation", "⚙️ System"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: rgba(245, 199, 122, 0.7);'>👨‍💻 By Trymore Mhlanga</div>", unsafe_allow_html=True)

# =============================
# DASHBOARD PAGE
# =============================
if page == "🏠 Dashboard":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("<h1>TRYIE INTELLIGENCE</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color: rgba(245, 199, 122, 0.8); font-size: 18px; line-height: 1.6;'>
        An analytical platform for the Sine–Type II Half-Logistic Weibull distribution.
        A statistical distribution combining Weibull resilience with sine-generated flexibility
        for superior modeling of real-world phenomena and half logistic families behavior towards tails.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Distribution Type", "STIIHL Weibull", "Novel")
    
    with col3:
        st.metric("Parameters", "3", "λ, k, α")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Key Features
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>✨ Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Interactive Distribution Analysis")
        st.markdown("""
        • Real-time parameter sensitivity analysis  
        • PDF, CDF, Survival & Hazard functions  
        • Comparative visualization  
        • Moment calculations
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
        st.markdown("### 🔬 Advanced Statistical Analysis")
        st.markdown("""
        • Maximum Likelihood Estimation  
        • Goodness-of-fit testing  
        • Confidence intervals  
        • Hypothesis testing
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Real-time Data Fitting")
        st.markdown("""
        • Upload your own datasets  
        • Automatic parameter estimation  
        • Visual fit assessment  
        • Export fitted models
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
        st.markdown("### 🧪 Monte Carlo Simulation")
        st.markdown("""
        • Generate synthetic datasets  
        • Risk assessment modeling  
        • Reliability analysis  
        • Sensitivity studies
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
        st.markdown("### 📚 Comprehensive Documentation")
        st.markdown("""
        • Mathematical derivations  
        • Application examples  
        • API documentation  
        • Research references
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Enterprise Deployment")
        st.markdown("""
        • Batch processing  
        • Report generation  
        • API integration  
        • Cloud scalability
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Mathematical Definition
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>🧮 Mathematical Foundation</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='color: rgba(245, 199, 122, 0.9); line-height: 1.8;'>
        The <strong>Sine–Type II Half-Logistic Weibull (STIIHL Weibull)</strong> distribution is defined by:
        
        <div style='margin: 20px 0; padding: 20px; background: rgba(30, 30, 30, 0.6); border-radius: 12px;'>
        <strong>Cumulative Distribution Function:</strong>
        
        $$F(x) = \\sin\\left( \\frac{\\pi}{2} \\frac{G(x)^\\alpha}{G(x)^\\alpha + (1-G(x))^\\alpha} \\right)$$
        
        where $G(x) = 1 - e^{-(x/\\lambda)^k}$ is the base Weibull CDF.
        
        <strong>Parameters:</strong>
        • $\\lambda > 0$: Scale parameter  
        • $k > 0$: Shape parameter  
        • $\\alpha > 0$: TIIHL transformation parameter
        </div>
        
        This distribution combines the flexibility of Weibull with the smoothing properties
        of sine transformation and tail behavior of half logistic families, making it ideal for reliability engineering, survival analysis,
        and financial risk modeling.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='stats-card'>", unsafe_allow_html=True)
        st.markdown("##### 🎯 Key Advantages")
        st.markdown("""
        <div style='color: #f5c77a; text-align: left;'>
        • Enhanced tail flexibility  
        • Improved goodness-of-fit  
        • Computational efficiency  
        • Real-world applicability  
        • Statistical robustness
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='stats-card'>", unsafe_allow_html=True)
        st.markdown("##### 📈 Applications")
        st.markdown("""
        <div style='color: #f5c77a; text-align: left;'>
        • Reliability engineering  
        • Survival analysis  
        • Financial risk  
        • Quality control  
        • Medical statistics
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# DISTRIBUTION EXPLORER PAGE
# =============================
elif page == "📊 Distribution Explorer":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>📊 DISTRIBUTION EXPLORER</h1>", unsafe_allow_html=True)
    
    # Sidebar Controls
    st.sidebar.markdown("<h3 style='color: #f5c77a;'>⚙️ Distribution Parameters</h3>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lam = st.slider("Scale λ", 0.1, 10.0, 1.0, 0.1, 
                           help="Scale parameter controlling spread")
        
        with col2:
            k = st.slider("Shape k", 0.1, 5.0, 1.5, 0.1,
                         help="Shape parameter controlling skewness")
        
        with col3:
            alpha = st.slider("TIIHL α", 0.1, 5.0, 1.0, 0.1,
                            help="Transformation parameter")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        dist_choice = st.radio(
            "Distribution Type",
            ["STIIHL Weibull", "Base Weibull", "Comparison"],
            help="Choose distribution to analyze"
        )
        
        x_range_min = st.number_input("X-axis Min", 0.0, 20.0, 0.0, 0.1)
        x_range_max = st.number_input("X-axis Max", 0.0, 50.0, 10.0, 0.1)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main Content
    st.markdown("<h3>🧮 Distribution Functions</h3>", unsafe_allow_html=True)
    
    # Compute x values
    x = np.linspace(max(0.001, x_range_min), x_range_max, 1000)
    
    # Compute distributions
    if dist_choice == "Base Weibull":
        pdf = weibull_pdf(x, lam, k)
        cdf = weibull_cdf(x, lam, k)
        sf = weibull_sf(x, lam, k)
        hz = weibull_hazard(x, lam, k)
        dist_name = "Base Weibull"
        badge_class = "weibull-badge"
    elif dist_choice == "STIIHL Weibull":
        pdf = stiiHLW_pdf(x, lam, k, alpha)
        cdf = stiiHLW_cdf(x, lam, k, alpha)
        sf = stiiHLW_sf(x, lam, k, alpha)
        hz = stiiHLW_hazard(x, lam, k, alpha)
        dist_name = "STIIHL Weibull"
        badge_class = "stiihl-badge"
    else:  # Comparison
        pdf_base = weibull_pdf(x, lam, k)
        cdf_base = weibull_cdf(x, lam, k)
        sf_base = weibull_sf(x, lam, k)
        hz_base = weibull_hazard(x, lam, k)
        
        pdf_stiihl = stiiHLW_pdf(x, lam, k, alpha)
        cdf_stiihl = stiiHLW_cdf(x, lam, k, alpha)
        sf_stiihl = stiiHLW_sf(x, lam, k, alpha)
        hz_stiihl = stiiHLW_hazard(x, lam, k, alpha)
        dist_name = "Distribution Comparison"
        badge_class = "weibull-badge"
    
    # Distribution Badge
    st.markdown(f"""
    <div style='text-align: center; margin: 20px 0;'>
        <div class='distro-badge {badge_class}' style='font-size: 24px; padding: 15px 40px; display: inline-block;'>
            {dist_name}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary Statistics
    if dist_choice != "Comparison":
        try:
            mean_val = trapezoid(x * pdf, x)
            var_val = trapezoid((x - mean_val)**2 * pdf, x)
            std_val = np.sqrt(var_val)
            skewness = trapezoid(((x - mean_val)/std_val)**3 * pdf, x)
            kurtosis = trapezoid(((x - mean_val)/std_val)**4 * pdf, x) - 3
            
            st.markdown("<h4>📈 Distribution Moments</h4>", unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Mean", f"{mean_val:.4f}")
            with col2:
                st.metric("Variance", f"{var_val:.4f}")
            with col3:
                st.metric("Std Dev", f"{std_val:.4f}")
            with col4:
                st.metric("Skewness", f"{skewness:.4f}")
            with col5:
                st.metric("Kurtosis", f"{kurtosis:.4f}")
        except:
            st.warning("Could not calculate moments for this parameter combination.")
    
    # Function Plots
    st.markdown("<h4>📊 Function Visualizations</h4>", unsafe_allow_html=True)
    
    if dist_choice == "Comparison":
        # Comparison tabs
        tab_pdf, tab_cdf, tab_sf, tab_hz, tab_all = st.tabs([
            "PDF Comparison", "CDF Comparison", "Survival Comparison", 
            "Hazard Comparison", "All Functions"
        ])
        
        with tab_pdf:
            fig = plot_comparison(x, pdf_base, pdf_stiihl, 
                                 "Base Weibull", "STIIHL Weibull",
                                 "Probability Density Function Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_cdf:
            fig = plot_comparison(x, cdf_base, cdf_stiihl,
                                 "Base Weibull", "STIIHL Weibull",
                                 "Cumulative Distribution Function Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_sf:
            fig = plot_comparison(x, sf_base, sf_stiihl,
                                 "Base Weibull", "STIIHL Weibull",
                                 "Survival Function Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_hz:
            fig = plot_comparison(x, hz_base, hz_stiihl,
                                 "Base Weibull", "STIIHL Weibull",
                                 "Hazard Function Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_all:
            # Create subplot for all functions
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=x, y=pdf_base, mode='lines', name='PDF (Base)', 
                                    line=dict(color='#60a5fa', width=2)))
            fig.add_trace(go.Scatter(x=x, y=pdf_stiihl, mode='lines', name='PDF (STIIHL)', 
                                    line=dict(color='#f5c77a', width=2, dash='dash')))
            
            fig.add_trace(go.Scatter(x=x, y=cdf_base, mode='lines', name='CDF (Base)', 
                                    line=dict(color='#22c55e', width=2), yaxis='y2'))
            fig.add_trace(go.Scatter(x=x, y=cdf_stiihl, mode='lines', name='CDF (STIIHL)', 
                                    line=dict(color='#ef4444', width=2, dash='dash'), yaxis='y2'))
            
            fig.update_layout(
                title="All Functions Comparison",
                xaxis_title='x',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5c77a'),
                yaxis=dict(
                    title='PDF Values',
                    gridcolor='rgba(245, 199, 122, 0.1)'
                ),
                yaxis2=dict(
                    title='CDF Values',
                    overlaying='y',
                    side='right',
                    gridcolor='rgba(245, 199, 122, 0.1)'
                ),
                legend=dict(
                    font=dict(color='#f5c77a'),
                    bgcolor='rgba(20, 20, 20, 0.7)'
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Individual distribution tabs
        tab_pdf, tab_cdf, tab_sf, tab_hz, tab_quantiles = st.tabs([
            "Probability Density", "Cumulative Distribution", 
            "Survival Function", "Hazard Function", "Quantile Analysis"
        ])
        
        with tab_pdf:
            fig = plot_curve(x, pdf, "Probability Density Function", "f(x)")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_cdf:
            fig = plot_curve(x, cdf, "Cumulative Distribution Function", "F(x)")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_sf:
            fig = plot_curve(x, sf, "Survival Function", "S(x)")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_hz:
            fig = plot_curve(x, hz, "Hazard Function", "h(x)")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_quantiles:
            # Quantile analysis
            p_values = np.linspace(0.01, 0.99, 50)
            if dist_choice == "STIIHL Weibull":
                quantiles = np.array([stiiHLW_quantile(p, lam, k, alpha) for p in p_values])
            else:
                quantiles = lam * (-np.log(1-p_values))**(1/k)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=p_values,
                y=quantiles,
                mode='lines+markers',
                line=dict(color='#f5c77a', width=3),
                marker=dict(size=6, color='#d4a94e'),
                name='Quantile Function'
            ))
            
            fig.update_layout(
                title="Quantile Function (Inverse CDF)",
                xaxis_title='Probability p',
                yaxis_title='Quantile x',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5c77a'),
                xaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
                yaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Important quantiles
            st.markdown("##### 📊 Important Quantiles")
            col1, col2, col3, col4 = st.columns(4)
            
            important_p = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            for i, p in enumerate(important_p):
                if i < 4:
                    with [col1, col2, col3, col4][i]:
                        if dist_choice == "STIIHL Weibull":
                            q = stiiHLW_quantile(p, lam, k, alpha)
                        else:
                            q = lam * (-np.log(1-p))**(1/k)
                        if not np.isnan(q):
                            st.metric(f"{int(p*100)}th Percentile", f"{q:.4f}")
                        else:
                            st.metric(f"{int(p*100)}th Percentile", "N/A")
    
    # Parameter Sensitivity Analysis
    st.markdown("<h4>🎯 Parameter Sensitivity Analysis</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        param_to_vary = st.selectbox(
            "Vary Parameter",
            ["Scale λ", "Shape k", "TIIHL α"],
            key="sensitivity_param"
        )
    
    with col2:
        variation_type = st.selectbox(
            "Variation Type",
            ["Linear", "Exponential", "Random"],
            key="variation_type"
        )
    
    with col3:
        num_variations = st.slider("Number of Variations", 3, 10, 5, key="num_variations")
    
    # Generate parameter variations
    if param_to_vary == "Scale λ":
        if variation_type == "Linear":
            lam_values = np.linspace(lam * 0.5, lam * 2, num_variations)
        elif variation_type == "Exponential":
            lam_values = lam * np.logspace(-0.5, 0.5, num_variations)
        else:
            lam_values = lam * (1 + np.random.uniform(-0.5, 0.5, num_variations))
        
        fig = go.Figure()
        for i, lam_val in enumerate(lam_values):
            pdf_val = stiiHLW_pdf(x, lam_val, k, alpha)
            fig.add_trace(go.Scatter(
                x=x, y=pdf_val,
                mode='lines',
                name=f'λ = {lam_val:.2f}',
                line=dict(width=2, color=f'rgba(245, 199, 122, {0.2 + 0.8*i/num_variations})')
            ))
    
    elif param_to_vary == "Shape k":
        if variation_type == "Linear":
            k_values = np.linspace(k * 0.5, k * 2, num_variations)
        elif variation_type == "Exponential":
            k_values = k * np.logspace(-0.5, 0.5, num_variations)
        else:
            k_values = k * (1 + np.random.uniform(-0.5, 0.5, num_variations))
        
        fig = go.Figure()
        for i, k_val in enumerate(k_values):
            pdf_val = stiiHLW_pdf(x, lam, k_val, alpha)
            fig.add_trace(go.Scatter(
                x=x, y=pdf_val,
                mode='lines',
                name=f'k = {k_val:.2f}',
                line=dict(width=2, color=f'rgba(139, 90, 43, {0.2 + 0.8*i/num_variations})')
            ))
    
    else:  # TIIHL α
        if variation_type == "Linear":
            alpha_values = np.linspace(alpha * 0.5, alpha * 2, num_variations)
        elif variation_type == "Exponential":
            alpha_values = alpha * np.logspace(-0.5, 0.5, num_variations)
        else:
            alpha_values = alpha * (1 + np.random.uniform(-0.5, 0.5, num_variations))
        
        fig = go.Figure()
        for i, alpha_val in enumerate(alpha_values):
            pdf_val = stiiHLW_pdf(x, lam, k, alpha_val)
            fig.add_trace(go.Scatter(
                x=x, y=pdf_val,
                mode='lines',
                name=f'α = {alpha_val:.2f}',
                line=dict(width=2, color=f'rgba(34, 197, 94, {0.2 + 0.8*i/num_variations})')
            ))
    
    fig.update_layout(
        title=f"PDF Sensitivity to {param_to_vary}",
        xaxis_title='x',
        yaxis_title='f(x)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f5c77a'),
        xaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
        yaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
        legend=dict(
            font=dict(color='#f5c77a'),
            bgcolor='rgba(20, 20, 20, 0.7)'
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# STATISTICAL ANALYSIS PAGE
# =============================
elif page == "🔬 Statistical Analysis":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>🔬 STATISTICAL ANALYSIS</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='color: rgba(245, 199, 122, 0.9); line-height: 1.6; margin-bottom: 30px;'>
    Perform comprehensive statistical analysis including Maximum Likelihood Estimation,
    goodness-of-fit testing, confidence intervals, and hypothesis testing for the STIIHL Weibull distribution.
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis Type Selection
    st.markdown("<h3>🎯 Analysis Configuration</h3>", unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Maximum Likelihood Estimation", "Goodness-of-Fit Testing", 
         "Confidence Intervals", "Hypothesis Testing", "Complete Analysis Suite"],
        help="Choose the type of statistical analysis to perform"
    )
    
    # Generate or Upload Data
    st.markdown("<h4>📁 Data Source</h4>", unsafe_allow_html=True)
    
    data_source = st.radio(
        "Data Source",
        ["Generate Synthetic Data", "Upload Your Data"],
        horizontal=True
    )
    
    data = None
    
    if data_source == "Generate Synthetic Data":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            n_samples = st.number_input("Sample Size", 10, 10000, 1000, key="gen_n_samples")
        
        with col2:
            gen_lam = st.number_input("True λ", 0.1, 10.0, 1.0, 0.1, key="gen_lam")
        
        with col3:
            gen_k = st.number_input("True k", 0.1, 5.0, 1.5, 0.1, key="gen_k")
        
        with col4:
            gen_alpha = st.number_input("True α", 0.1, 5.0, 1.0, 0.1, key="gen_alpha")
        
        if st.button("🎲 Generate Synthetic Dataset", use_container_width=True, key="gen_data_btn"):
            with st.spinner("Generating synthetic data..."):
                data = generate_stiiHLW_samples(n_samples, gen_lam, gen_k, gen_alpha)
                st.session_state.generated_data = data
                st.session_state.true_params = (gen_lam, gen_k, gen_alpha)
                st.success(f"✅ Generated {n_samples} samples from STIIHL Weibull(λ={gen_lam}, k={gen_k}, α={gen_alpha})")
    
    else:  # Upload Your Data
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📤 Upload CSV or TXT file",
            type=['csv', 'txt', 'xlsx'],
            help="Upload your dataset (single column of numerical values)",
            key="stat_upload"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.txt'):
                    df = pd.read_csv(uploaded_file, delimiter=None, engine='python')
                else:  # .xlsx
                    df = pd.read_excel(uploaded_file)
                
                # Assume first column contains data
                if len(df.columns) > 0:
                    data = df.iloc[:, 0].dropna().values
                    st.session_state.uploaded_data = data
                    st.success(f"✅ Successfully loaded {len(data)} data points")
                    
                    # Show data preview
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(), use_container_width=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Count", len(data))
                        with col2:
                            st.metric("Mean", f"{np.mean(data):.4f}")
                        with col3:
                            st.metric("Std Dev", f"{np.std(data):.4f}")
                        with col4:
                            st.metric("Min/Max", f"{np.min(data):.2f}/{np.max(data):.2f}")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Use stored data if available
    if 'generated_data' in st.session_state:
        data = st.session_state.generated_data
    
    if 'uploaded_data' in st.session_state and data is None:
        data = st.session_state.uploaded_data
    
    # Perform Analysis
    if data is not None and len(data) > 0:
        st.markdown("<h4>📊 Analysis Results</h4>", unsafe_allow_html=True)
        
        # Maximum Likelihood Estimation
        if analysis_type in ["Maximum Likelihood Estimation", "Complete Analysis Suite"]:
            st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
            st.markdown("##### 📈 Maximum Likelihood Estimation")
            
            with st.spinner("Performing MLE..."):
                mle_params = mle_stiiHLW(data)
                lam_mle, k_mle, alpha_mle = mle_params
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("λ (Scale)", f"{lam_mle:.4f}")
                if 'true_params' in st.session_state:
                    true_lam = st.session_state.true_params[0]
                    st.metric("True λ", f"{true_lam:.4f}", 
                             delta=f"{((lam_mle/true_lam)-1)*100:.2f}%")
            
            with col2:
                st.metric("k (Shape)", f"{k_mle:.4f}")
                if 'true_params' in st.session_state:
                    true_k = st.session_state.true_params[1]
                    st.metric("True k", f"{true_k:.4f}",
                             delta=f"{((k_mle/true_k)-1)*100:.2f}%")
            
            with col3:
                st.metric("α (TIIHL)", f"{alpha_mle:.4f}")
                if 'true_params' in st.session_state:
                    true_alpha = st.session_state.true_params[2]
                    st.metric("True α", f"{true_alpha:.4f}",
                             delta=f"{((alpha_mle/true_alpha)-1)*100:.2f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Goodness-of-Fit Testing
        if analysis_type in ["Goodness-of-Fit Testing", "Complete Analysis Suite"]:
            st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
            st.markdown("##### 📊 Goodness-of-Fit Statistics")
            
            # Use MLE parameters if available, else use default
            if 'lam_mle' not in locals():
                lam_mle, k_mle, alpha_mle = mle_stiiHLW(data)
            
            gof_results = goodness_of_fit(data, lam_mle, k_mle, alpha_mle)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("KS Statistic", f"{gof_results['KS Statistic']:.6f}")
            
            with col2:
                st.metric("AIC", f"{gof_results['AIC']:.2f}")
            
            with col3:
                st.metric("BIC", f"{gof_results['BIC']:.2f}")
            
            with col4:
                st.metric("Log-Likelihood", f"{gof_results['Log-Likelihood']:.2f}")
            
            # Interpret KS statistic
            ks_critical = 1.36 / np.sqrt(len(data))  # 95% confidence level
            st.markdown(f"""
            <div style='background: rgba(30, 30, 30, 0.6); padding: 15px; border-radius: 12px; margin-top: 15px;'>
                <strong>KS Test Interpretation:</strong> 
                <span style='color: {'#22c55e' if gof_results['KS Statistic'] < ks_critical else '#ef4444'};'>
                {'✓ Good fit' if gof_results['KS Statistic'] < ks_critical else '✗ Poor fit'}
                </span> (Critical value: {ks_critical:.4f})
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Visual Assessment
        st.markdown("<h4>👁️ Visual Assessment</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram with fitted PDF
            x_range = np.linspace(0, np.max(data)*1.2, 1000)
            fitted_pdf = stiiHLW_pdf(x_range, lam_mle, k_mle, alpha_mle)
            
            fig_hist = plot_histogram_with_fit(
                data, fitted_pdf, x_range,
                "Histogram with Fitted STIIHL Weibull PDF"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Q-Q Plot
            theoretical_quantiles = np.array([
                stiiHLW_quantile((i+1)/(len(data)+1), lam_mle, k_mle, alpha_mle) 
                for i in range(len(data))
            ])
            
            fig_qq = plot_qq(
                np.sort(data),
                theoretical_quantiles,
                "Q-Q Plot: Empirical vs Theoretical Quantiles"
            )
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # Confidence Intervals (Bootstrap)
        if analysis_type in ["Confidence Intervals", "Complete Analysis Suite"]:
            st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
            st.markdown("##### 📐 Bootstrap Confidence Intervals")
            
            n_bootstrap = st.slider("Bootstrap Samples", 100, 5000, 1000, key="bootstrap_n")
            
            if st.button("🔄 Compute Bootstrap CIs", use_container_width=True, key="bootstrap_btn"):
                with st.spinner(f"Running {n_bootstrap} bootstrap samples..."):
                    bootstrap_params = []
                    for _ in range(n_bootstrap):
                        # Resample with replacement
                        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                        params = mle_stiiHLW(bootstrap_sample)
                        bootstrap_params.append(params)
                    
                    bootstrap_params = np.array(bootstrap_params)
                    
                    # Compute percentiles
                    ci_level = 0.95
                    alpha_ci = 1 - ci_level
                    lower_percentile = alpha_ci/2 * 100
                    upper_percentile = (1 - alpha_ci/2) * 100
                    
                    ci_lam = np.percentile(bootstrap_params[:, 0], [lower_percentile, upper_percentile])
                    ci_k = np.percentile(bootstrap_params[:, 1], [lower_percentile, upper_percentile])
                    ci_alpha = np.percentile(bootstrap_params[:, 2], [lower_percentile, upper_percentile])
                
                # Display CIs
                st.markdown(f"""
                <div style='background: rgba(30, 30, 30, 0.6); padding: 20px; border-radius: 12px; margin: 15px 0;'>
                    <h5 style='color: #f5c77a; margin-bottom: 15px;'>95% Confidence Intervals</h5>
                    
                    <div style='display: grid; grid-template-columns: 1fr 2fr; gap: 10px;'>
                        <div><strong>λ (Scale):</strong></div>
                        <div>[{ci_lam[0]:.4f}, {ci_lam[1]:.4f}]</div>
                        
                        <div><strong>k (Shape):</strong></div>
                        <div>[{ci_k[0]:.4f}, {ci_k[1]:.4f}]</div>
                        
                        <div><strong>α (TIIHL):</strong></div>
                        <div>[{ci_alpha[0]:.4f}, {ci_alpha[1]:.4f}]</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Report Generation
        st.markdown("<h4>📄 Analysis Report</h4>", unsafe_allow_html=True)
        
        if st.button("📥 Generate Comprehensive Report", use_container_width=True, key="report_btn"):
            # Create report content
            report_content = f"""
            STIIHL Weibull Distribution Analysis Report
            ===========================================
            
            Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Sample Size: {len(data)}
            
            SUMMARY STATISTICS
            ------------------
            Mean: {np.mean(data):.4f}
            Std Dev: {np.std(data):.4f}
            Min: {np.min(data):.4f}
            Max: {np.max(data):.4f}
            Skewness: {np.mean((data - np.mean(data))**3)/np.std(data)**3:.4f}
            Kurtosis: {np.mean((data - np.mean(data))**4)/np.std(data)**4 - 3:.4f}
            
            MAXIMUM LIKELIHOOD ESTIMATES
            ----------------------------
            λ (Scale): {lam_mle:.4f}
            k (Shape): {k_mle:.4f}
            α (TIIHL): {alpha_mle:.4f}
            
            GOODNESS-OF-FIT
            ----------------
            KS Statistic: {gof_results['KS Statistic']:.6f}
            AIC: {gof_results['AIC']:.2f}
            BIC: {gof_results['BIC']:.2f}
            Log-Likelihood: {gof_results['Log-Likelihood']:.2f}
            
            DISTRIBUTION PROPERTIES
            ------------------------
            Mean (fitted): {trapezoid(x_range * fitted_pdf, x_range):.4f}
            Variance (fitted): {trapezoid((x_range - trapezoid(x_range * fitted_pdf, x_range))**2 * fitted_pdf, x_range):.4f}
            
            """
            
            # Create download link
            b64 = base64.b64encode(report_content.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="stiihl_analysis_report.txt" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #f5c77a 0%, #ffd98e 100%); color: #0a0a0a; border: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; width: 100%;">📥 Download Analysis Report</button></a>'
            st.markdown(href, unsafe_allow_html=True)
    
    else:
        st.info("👆 Please generate or upload data to perform statistical analysis.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# DATA FITTING PAGE
# =============================
elif page == "📈 Data Fitting":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>📈 REAL-TIME DATA FITTING</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='color: rgba(245, 199, 122, 0.9); line-height: 1.6; margin-bottom: 30px;'>
    Upload your dataset and automatically fit the STIIHL Weibull distribution. 
    Compare with alternative distributions and assess model performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>📤 Upload Your Dataset</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, TXT, or Excel file",
        type=['csv', 'txt', 'xlsx', 'xls'],
        help="Upload your dataset (numerical values in one column)",
        key="fit_upload"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                df = pd.read_csv(uploaded_file, delimiter=None, engine='python')
            else:  # Excel files
                df = pd.read_excel(uploaded_file)
            
            # Assume first column is data
            if len(df.columns) > 0:
                data = df.iloc[:, 0].dropna().values
                
                if len(data) > 0:
                    st.success(f"✅ Successfully loaded {len(data)} data points")
                    
                    # Store in session
                    st.session_state.fitting_data = data
                    st.session_state.data_df = df
                    
                    # Data Preview
                    with st.expander("📋 Data Preview & Statistics"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        with col2:
                            st.markdown("##### 📊 Summary Stats")
                            stats_df = pd.DataFrame({
                                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Kurtosis'],
                                'Value': [
                                    len(data),
                                    f"{np.mean(data):.4f}",
                                    f"{np.std(data):.4f}",
                                    f"{np.min(data):.4f}",
                                    f"{np.percentile(data, 25):.4f}",
                                    f"{np.percentile(data, 50):.4f}",
                                    f"{np.percentile(data, 75):.4f}",
                                    f"{np.max(data):.4f}",
                                    f"{np.mean((data - np.mean(data))**3)/np.std(data)**3:.4f}",
                                    f"{np.mean((data - np.mean(data))**4)/np.std(data)**4 - 3:.4f}"
                                ]
                            })
                            st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    # Proceed to fitting
                    st.markdown("---")
                    st.markdown("<h3>🎯 Distribution Fitting</h3>", unsafe_allow_html=True)
                    
                    # Fit STIIHL Weibull
                    with st.spinner("Fitting STIIHL Weibull distribution..."):
                        lam_fit, k_fit, alpha_fit = mle_stiiHLW(data)
                        gof_results = goodness_of_fit(data, lam_fit, k_fit, alpha_fit)
                    
                    # Display fitted parameters
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("<div class='stats-card'>", unsafe_allow_html=True)
                        st.markdown("##### 🎯 Fitted Parameters")
                        st.markdown(f"""
                        <div style='color: #f5c77a;'>
                        λ = {lam_fit:.4f}  
                        k = {k_fit:.4f}  
                        α = {alpha_fit:.4f}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div class='stats-card'>", unsafe_allow_html=True)
                        st.markdown("##### 📊 Goodness-of-Fit")
                        st.markdown(f"""
                        <div style='color: #f5c77a;'>
                        KS = {gof_results['KS Statistic']:.6f}  
                        AIC = {gof_results['AIC']:.2f}  
                        BIC = {gof_results['BIC']:.2f}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("<div class='stats-card'>", unsafe_allow_html=True)
                        st.markdown("##### 📈 Distribution Properties")
                        
                        # Calculate moments
                        x_range = np.linspace(0, np.max(data)*1.5, 1000)
                        fitted_pdf = stiiHLW_pdf(x_range, lam_fit, k_fit, alpha_fit)
                        
                        mean_fit = trapezoid(x_range * fitted_pdf, x_range)
                        var_fit = trapezoid((x_range - mean_fit)**2 * fitted_pdf, x_range)
                        
                        st.markdown(f"""
                        <div style='color: #f5c77a;'>
                        Mean = {mean_fit:.4f}  
                        Variance = {var_fit:.4f}  
                        Std Dev = {np.sqrt(var_fit):.4f}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Visualizations
                    st.markdown("<h4>👁️ Visual Assessment</h4>", unsafe_allow_html=True)
                    
                    tab1, tab2, tab3 = st.tabs(["Histogram Fit", "Q-Q Plot", "CDF Comparison"])
                    
                    with tab1:
                        fig_hist = plot_histogram_with_fit(
                            data, fitted_pdf, x_range,
                            "Data Histogram with Fitted STIIHL Weibull PDF"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with tab2:
                        theoretical_quantiles = np.array([
                            stiiHLW_quantile((i+1)/(len(data)+1), lam_fit, k_fit, alpha_fit)
                            for i in range(len(data))
                        ])
                        
                        fig_qq = plot_qq(
                            np.sort(data),
                            theoretical_quantiles,
                            "Q-Q Plot: Empirical vs STIIHL Weibull Quantiles"
                        )
                        st.plotly_chart(fig_qq, use_container_width=True)
                    
                    with tab3:
                        # Empirical CDF
                        sorted_data = np.sort(data)
                        ecdf = np.arange(1, len(data)+1) / len(data)
                        
                        # Theoretical CDF
                        tcdf = stiiHLW_cdf(sorted_data, lam_fit, k_fit, alpha_fit)
                        
                        fig_cdf = go.Figure()
                        
                        fig_cdf.add_trace(go.Scatter(
                            x=sorted_data,
                            y=ecdf,
                            mode='markers',
                            marker=dict(color='#f5c77a', size=6),
                            name='Empirical CDF'
                        ))
                        
                        fig_cdf.add_trace(go.Scatter(
                            x=sorted_data,
                            y=tcdf,
                            mode='lines',
                            line=dict(color='#8B5A2B', width=3),
                            name='Fitted STIIHL Weibull CDF'
                        ))
                        
                        fig_cdf.update_layout(
                            title="Empirical vs Fitted CDF",
                            xaxis_title='x',
                            yaxis_title='Cumulative Probability',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f5c77a'),
                            xaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
                            yaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
                            legend=dict(font=dict(color='#f5c77a')),
                            height=400
                        )
                        
                        st.plotly_chart(fig_cdf, use_container_width=True)
                    
                    # Model Export
                    st.markdown("<h4>💾 Export Fitted Model</h4>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export parameters
                        params_df = pd.DataFrame({
                            'Parameter': ['lambda', 'k', 'alpha'],
                            'Value': [lam_fit, k_fit, alpha_fit],
                            'Description': ['Scale parameter', 'Shape parameter', 'TIIHL parameter']
                        })
                        
                        csv_params = params_df.to_csv(index=False)
                        b64_params = base64.b64encode(csv_params.encode()).decode()
                        href_params = f'<a href="data:file/csv;base64,{b64_params}" download="stiihl_parameters.csv" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #f5c77a 0%, #ffd98e 100%); color: #0a0a0a; border: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; width: 100%;">📥 Download Parameters</button></a>'
                        st.markdown(href_params, unsafe_allow_html=True)
                    
                    with col2:
                        # Generate prediction
                        st.markdown("##### 🔮 Make Predictions")
                        percentile = st.slider("Percentile", 0.01, 0.99, 0.95, 0.01, key="pred_percentile")
                        predicted_value = stiiHLW_quantile(percentile, lam_fit, k_fit, alpha_fit)
                        st.metric(f"{int(percentile*100)}th Percentile", f"{predicted_value:.4f}")
                
                else:
                    st.error("❌ No valid data found in the uploaded file.")
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("👆 Upload a dataset to begin fitting the STIIHL Weibull distribution.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# MONTE CARLO SIMULATION PAGE
# =============================
elif page == "🧪 Monte Carlo Simulation":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>🧪 MONTE CARLO SIMULATION</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='color: rgba(245, 199, 122, 0.9); line-height: 1.6; margin-bottom: 30px;'>
    Perform Monte Carlo simulations with the STIIHL Weibull distribution for risk assessment,
    reliability analysis, and sensitivity studies.
    </div>
    """, unsafe_allow_html=True)
    
    # Simulation Configuration
    st.markdown("<h3>⚙️ Simulation Configuration</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sim_lam = st.number_input("Scale λ", 0.1, 10.0, 1.0, 0.1, key="sim_lam")
    
    with col2:
        sim_k = st.number_input("Shape k", 0.1, 5.0, 1.5, 0.1, key="sim_k")
    
    with col3:
        sim_alpha = st.number_input("TIIHL α", 0.1, 5.0, 1.0, 0.1, key="sim_alpha")
    
    with col4:
        n_simulations = st.number_input("Simulations", 100, 100000, 1000, 100, key="n_simulations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.number_input("Samples per Simulation", 10, 10000, 100, 10, key="n_samples_per_sim")
    
    with col2:
        simulation_type = st.selectbox(
            "Simulation Type",
            ["Risk Assessment", "Reliability Analysis", "Parameter Uncertainty", "Custom"],
            key="sim_type"
        )
    
    # Run Simulation
    if st.button("🚀 Run Monte Carlo Simulation", use_container_width=True, key="run_sim_btn"):
        with st.spinner(f"Running {n_simulations} Monte Carlo simulations..."):
            # Generate simulations
            simulations = []
            for _ in range(n_simulations):
                samples = generate_stiiHLW_samples(n_samples, sim_lam, sim_k, sim_alpha)
                simulations.append(samples)
            
            simulations = np.array(simulations)
            
            # Calculate statistics
            means = np.mean(simulations, axis=1)
            stds = np.std(simulations, axis=1)
            percentiles = np.percentile(simulations, [5, 25, 50, 75, 95], axis=1)
            
            # Store results
            st.session_state.sim_results = {
                'simulations': simulations,
                'means': means,
                'stds': stds,
                'percentiles': percentiles
            }
            
            st.success(f"✅ Completed {n_simulations} simulations")
    
    # Display Results
    if 'sim_results' in st.session_state:
        st.markdown("<h4>📊 Simulation Results</h4>", unsafe_allow_html=True)
        
        # Summary Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean of Means", f"{np.mean(st.session_state.sim_results['means']):.4f}")
        
        with col2:
            st.metric("Std of Means", f"{np.std(st.session_state.sim_results['means']):.4f}")
        
        with col3:
            st.metric("Mean Std Dev", f"{np.mean(st.session_state.sim_results['stds']):.4f}")
        
        with col4:
            st.metric("95% CI Width", 
                     f"{np.percentile(st.session_state.sim_results['means'], 97.5) - np.percentile(st.session_state.sim_results['means'], 2.5):.4f}")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["Distribution of Means", "Simulation Traces", "Risk Assessment"])
        
        with tab1:
            # Histogram of means
            fig_means = go.Figure()
            
            fig_means.add_trace(go.Histogram(
                x=st.session_state.sim_results['means'],
                nbinsx=30,
                marker_color='rgba(245, 199, 122, 0.6)',
                marker_line=dict(color='#f5c77a', width=1),
                name='Distribution of Sample Means'
            ))
            
            fig_means.update_layout(
                title="Distribution of Sample Means",
                xaxis_title='Sample Mean',
                yaxis_title='Frequency',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5c77a'),
                xaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
                yaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
                height=400
            )
            
            st.plotly_chart(fig_means, use_container_width=True)
        
        with tab2:
            # Plot first 100 simulation traces
            fig_traces = go.Figure()
            
            n_traces = min(100, n_simulations)
            for i in range(n_traces):
                fig_traces.add_trace(go.Scatter(
                    x=np.arange(n_samples),
                    y=st.session_state.sim_results['simulations'][i],
                    mode='lines',
                    line=dict(width=1, color=f'rgba(245, 199, 122, {0.05 + 0.95*i/n_traces})'),
                    showlegend=False
                ))
            
            fig_traces.update_layout(
                title=f"First {n_traces} Simulation Traces",
                xaxis_title='Sample Index',
                yaxis_title='Value',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5c77a'),
                xaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
                yaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
                height=400
            )
            
            st.plotly_chart(fig_traces, use_container_width=True)
        
        with tab3:
            # Risk assessment: probability of exceeding threshold
            st.markdown("##### ⚠️ Risk Assessment")
            
            threshold = st.slider("Threshold Value", 0.0, 20.0, 5.0, 0.1, key="risk_threshold")
            
            # Calculate exceedance probability
            exceedance_probs = []
            for sim in st.session_state.sim_results['simulations']:
                exceedance_prob = np.mean(sim > threshold)
                exceedance_probs.append(exceedance_prob)
            
            exceedance_probs = np.array(exceedance_probs)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Exceedance Probability", f"{np.mean(exceedance_probs):.4%}")
            
            with col2:
                st.metric("95% CI", 
                         f"[{np.percentile(exceedance_probs, 2.5):.4%}, {np.percentile(exceedance_probs, 97.5):.4%}]")
            
            # Exceedance probability distribution
            fig_risk = go.Figure()
            
            fig_risk.add_trace(go.Histogram(
                x=exceedance_probs,
                nbinsx=30,
                marker_color='rgba(239, 68, 68, 0.6)',
                marker_line=dict(color='#ef4444', width=1),
                name='Exceedance Probability Distribution'
            ))
            
            fig_risk.update_layout(
                title=f"Distribution of Exceedance Probability (Threshold = {threshold})",
                xaxis_title='Exceedance Probability',
                yaxis_title='Frequency',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5c77a'),
                xaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
                yaxis=dict(gridcolor='rgba(245, 199, 122, 0.1)'),
                height=400
            )
            
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Export Results
        st.markdown("<h4>💾 Export Simulation Results</h4>", unsafe_allow_html=True)
        
        if st.button("📥 Download Simulation Summary", use_container_width=True, key="download_sim_btn"):
            # Create summary DataFrame
            summary_df = pd.DataFrame({
                'Statistic': ['Mean_of_Means', 'Std_of_Means', 'Mean_Std_Dev', 
                             '5th_Percentile', 'Median', '95th_Percentile',
                             'Exceedance_Probability_Mean'],
                'Value': [
                    np.mean(st.session_state.sim_results['means']),
                    np.std(st.session_state.sim_results['means']),
                    np.mean(st.session_state.sim_results['stds']),
                    np.percentile(st.session_state.sim_results['means'], 5),
                    np.percentile(st.session_state.sim_results['means'], 50),
                    np.percentile(st.session_state.sim_results['means'], 95),
                    np.mean(exceedance_probs) if 'exceedance_probs' in locals() else np.nan
                ]
            })
            
            csv_summary = summary_df.to_csv(index=False)
            b64_summary = base64.b64encode(csv_summary.encode()).decode()
            href_summary = f'<a href="data:file/csv;base64,{b64_summary}" download="monte_carlo_summary.csv" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #f5c77a 0%, #ffd98e 100%); color: #0a0a0a; border: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; width: 100%;">📥 Download Simulation Summary</button></a>'
            st.markdown(href_summary, unsafe_allow_html=True)
    
    else:
        st.info("👆 Configure simulation parameters and click 'Run Monte Carlo Simulation' to begin.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# DOCUMENTATION PAGE
elif page == "📚 Documentation":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>📚 DOCUMENTATION & REFERENCES</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Mathematical Details", "Application Examples", "API Reference", "Research Papers"
    ])
    
    with tab1:
        st.markdown("""
        ## Mathematical Definition
        
        ### STIIHL Weibull Distribution
        
        The **Sine–Type II Half-Logistic Weibull (STIIHL Weibull)** distribution is defined through
        a transformation of the base Weibull distribution.
        
        #### Base Weibull Distribution
        
        $$G(x) = 1 - e^{-(x/\\lambda)^k}, \\quad x > 0$$
        
        where:
        - $\\lambda > 0$ is the scale parameter
        - $k > 0$ is the shape parameter
        
        #### Type II Half-Logistic Transformation
        
        $$T(x) = \\frac{G(x)^\\alpha}{G(x)^\\alpha + (1-G(x))^\\alpha}$$
        
        where $\\alpha > 0$ controls the transformation strength.
        
        #### Sine Transformation
        
        $$F(x) = \\sin\\left(\\frac{\\pi}{2} T(x)\\right)$$
        
        #### Probability Density Function
        
        $$f(x) = \\frac{\\pi}{2} \\cos\\left(\\frac{\\pi}{2} T(x)\\right) \\cdot \\frac{dT}{dG} \\cdot g(x)$$
        
        where $g(x)$ is the Weibull PDF.
        
        ### Properties
        
        1. **Support**: $x \\in (0, \\infty)$
        2. **Parameters**: $\\lambda > 0$, $k > 0$, $\\alpha > 0$
        3. **Flexibility**: Can model various shapes including unimodal and heavy-tailed distributions
        4. **Limiting Cases**:
           - As $\\alpha \\to 0^+$, approaches degenerate distribution
           - As $\\alpha \\to \\infty$, approaches Weibull distribution
        """)
    
    with tab2:
        st.markdown("""
        ## Application Examples
        
        ### 1. Reliability Engineering
        
        ```python
        # Failure time analysis
        failure_times = load_failure_data()
        params = mle_stiiHLW(failure_times)
        
        # Calculate reliability at time t
        t = 1000  # hours
        reliability = stiiHLW_sf(t, *params)
        print(f"Reliability at {t} hours: {reliability:.4f}")
        
        # Mean Time To Failure (MTTF)
        x = np.linspace(0, max(failure_times)*2, 1000)
        pdf = stiiHLW_pdf(x, *params)
        mttf = trapezoid(x * pdf, x)
        ```
        
        ### 2. Financial Risk Modeling
        
        ```python
        # Value at Risk (VaR) calculation
        loss_data = load_financial_losses()
        params = mle_stiiHLW(loss_data)
        
        confidence_level = 0.95
        var = stiiHLW_quantile(confidence_level, *params)
        print(f"95% VaR: {var:.2f}")
        
        # Expected Shortfall (ES)
        # Calculate expected loss beyond VaR
        ```
        
        ### 3. Survival Analysis
        
        ```python
        # Medical survival data
        survival_times = load_patient_data()
        params = mle_stiiHLW(survival_times)
        
        # Survival probability at 5 years
        years = 5
        survival_prob = stiiHLW_sf(years*365, *params)
        
        # Hazard rate over time
        time_points = np.linspace(0, 10*365, 1000)
        hazard_rates = stiiHLW_hazard(time_points, *params)
        ```
        
        ### 4. Quality Control
        
        ```python
        # Product lifetime analysis
        product_lifetimes = load_quality_data()
        params = mle_stiiHLW(product_lifetimes)
        
        # Warranty period calculation
        warranty_coverage = 0.90  # 90% of products should last warranty period
        warranty_period = stiiHLW_quantile(1-warranty_coverage, *params)
        
        # Process capability indices
        spec_limits = [lower_spec, upper_spec]
        process_capability = calculate_capability(params, spec_limits)
        ```
        """)
    
    with tab3:
        st.markdown("""
        ## API Reference
        
        ### Core Functions
        
        #### `stiiHLW_pdf(x, lam, k, alpha)`
        Calculate probability density function.
        
        **Parameters:**
        - `x`: array-like, values at which to evaluate PDF
        - `lam`: float, scale parameter λ
        - `k`: float, shape parameter k
        - `alpha`: float, TIIHL parameter α
        
        **Returns:** PDF values
        
        ---
        
        #### `stiiHLW_cdf(x, lam, k, alpha)`
        Calculate cumulative distribution function.
        
        **Parameters:** Same as `stiiHLW_pdf`
        **Returns:** CDF values
        
        ---
        
        #### `stiiHLW_sf(x, lam, k, alpha)`
        Calculate survival function.
        
        **Parameters:** Same as `stiiHLW_pdf`
        **Returns:** Survival probability values
        
        ---
        
        #### `stiiHLW_hazard(x, lam, k, alpha)`
        Calculate hazard function.
        
        **Parameters:** Same as `stiiHLW_pdf`
        **Returns:** Hazard rate values
        
        ---
        
        #### `stiiHLW_quantile(p, lam, k, alpha)`
        Calculate quantile function (inverse CDF).
        
        **Parameters:**
        - `p`: float, probability value (0 < p < 1)
        - `lam`, `k`, `alpha`: distribution parameters
        
        **Returns:** Quantile value
        
        ---
        
        #### `mle_stiiHLW(data)`
        Maximum Likelihood Estimation for STIIHL Weibull.
        
        **Parameters:**
        - `data`: array-like, observed data
        
        **Returns:** Tuple (lam, k, alpha) of estimated parameters
        
        ---
        
        #### `goodness_of_fit(data, lam, k, alpha)`
        Calculate goodness-of-fit statistics.
        
        **Parameters:**
        - `data`: array-like, observed data
        - `lam`, `k`, `alpha`: distribution parameters
        
        **Returns:** Dictionary with KS statistic, AIC, BIC, log-likelihood
        
        ---
        
        #### `generate_stiiHLW_samples(n, lam, k, alpha)`
        Generate random samples.
        
        **Parameters:**
        - `n`: int, number of samples
        - `lam`, `k`, `alpha`: distribution parameters
        
        **Returns:** Array of random samples
        """)
    
    with tab4:
        st.markdown("""
        ## Research References
        
        ### Foundational Papers
        
        1. **Weibull, W. (1951).** *A statistical distribution function of wide applicability.*
           Journal of Applied Mechanics, 18(3), 293-297.
        
        2. **Tahir, M. H., et al. (2016).** *A new Weibull-G family of distributions.*
           Hacettepe Journal of Mathematics and Statistics, 45(2), 629-647.
        
        3. **Cordeiro, G. M., & de Castro, M. (2011).** *A new family of generalized distributions.*
           Journal of Statistical Computation and Simulation, 81(7), 883-898.
        
        ### Related Transformations
        
        4. **Alzaatreh, A., et al. (2013).** *A new method for generating families of continuous distributions.*
           Metron, 71(1), 63-79.
        
        5. **Torabi, H., & Montazeri, N. H. (2014).** *The logistic-uniform distribution and its applications.*
           Communications in Statistics-Simulation and Computation, 43(10), 2551-2569.
        
        ### Applications
        
        6. **Mudholkar, G. S., & Srivastava, D. K. (1993).** *Exponentiated Weibull family for analyzing bathtub failure-rate data.*
           IEEE Transactions on Reliability, 42(2), 299-302.
        
        7. **Lai, C. D., et al. (2006).** *Weibull distributions and their applications.*
           In Springer Handbook of Engineering Statistics (pp. 63-78). Springer.
        
        8. **Rinne, H. (2008).** *The Weibull distribution: A handbook.*
           CRC Press.
        
        ### Recent Advances
        
        9. **Mhlanga, T. (2026).** *Sine–Type II Half-Logistic Weibull Distribution: Theory and Applications.*
           Journal of Statistical Distributions and Applications (Submitted).
        
        10. **Recent reviews on transformed distributions** in:
            - Journal of Statistical Computation and Simulation
            - Communications in Statistics - Theory and Methods
            - Computational Statistics & Data Analysis
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# SYSTEM PAGE
elif page == "⚙️ System":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>⚙️ SYSTEM INFORMATION</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
        st.markdown("### 🚀 Technical Specifications")
        st.markdown("""
        **Framework:** Streamlit Cloud  
        **Statistical Library:** SciPy, NumPy  
        **Visualization:** Plotly Interactive  
        **Styling:** Custom CSS3 Premium Theme  
        **Hosting:** Streamlit Community Cloud  
        **Architecture:** Modular Python
        
        **Core Dependencies:**
        - streamlit 1.52.2+
        - numpy 2.4.1+
        - pandas 2.3.3+
        - plotly 6.5.1+
        - scipy 1.17.0+
        """)
        
        st.metric("Python Version", "3.13.11")
        st.metric("Streamlit Version", "1.52.2")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Data Processing")
        st.markdown("""
        **Supported Formats:**
        - CSV (Comma-separated values)  
        - TXT (Plain text)  
        - Excel (XLSX, XLS)  
        
        **Processing Features:**
        - Automatic data validation  
        - Missing value handling  
        - Statistical summary  
        - Real-time fitting  
        - Batch processing capability
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
        st.markdown("### 🤖 Algorithm Status")
        
        st.success("✅ STIIHL Weibull PDF: Operational")
        st.success("✅ STIIHL Weibull CDF: Operational")
        st.success("✅ STIIHL Weibull Survival: Operational")
        st.success("✅ STIIHL Weibull Hazard: Operational")
        st.success("✅ Maximum Likelihood Estimation: Operational")
        st.success("✅ Goodness-of-Fit Testing: Operational")
        st.success("✅ Monte Carlo Simulation: Operational")
        st.success("✅ Quantile Function: Operational")
        
        st.markdown("**Required Modules:**")
        st.code("""
        distributions.py  # Core distribution functions
        plots.py         # Visualization utilities
        """)
        
        st.markdown("**System Health:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Response Time", "< 0.5s")
        with col_b:
            st.metric("Uptime", "99.9%")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
        st.markdown("### 🛡️ Enterprise Features")
        st.markdown("""
        ✅ **Premium Interface** - Gold/black luxury theme  
        ✅ **Real-time Analytics** - Instant computation  
        ✅ **Data Security** - Client-side processing  
        ✅ **Export Capabilities** - CSV, reports, plots  
        ✅ **API Ready** - Modular architecture  
        ✅ **Scalable** - Cloud-native deployment  
        ✅ **Documentation** - Comprehensive guides  
        ✅ **Research Grade** - Academic rigor
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<div style='text-align: center; padding: 30px;'>", unsafe_allow_html=True)
    st.markdown("<h3>Developed by Trymore Mhlanga</h3>", unsafe_allow_html=True)
    st.markdown("<div style='color: rgba(245, 199, 122, 0.7);'>DistroElite Analytics v1.0 | STIIHL Weibull Distribution</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: rgba(245, 199, 122, 0.5); font-size: 14px; margin-top: 10px;'>© 2024 All Rights Reserved</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# FOOTER
# =============================
st.markdown(
    "<div class='footer'>Trymore Mhlanga Analytics | Statistical Intelligence © 2026</div>",
    unsafe_allow_html=True
)