import numpy as np

# =========================
# 1. BASE WEIBULL
# =========================

def weibull_cdf(x, lam, k):
    return 1 - np.exp(-(x / lam) ** k)

def weibull_pdf(x, lam, k):
    return (k / lam) * (x / lam) ** (k - 1) * np.exp(-(x / lam) ** k)

def weibull_sf(x, lam, k):
    return np.exp(-(x / lam) ** k)

def weibull_hazard(x, lam, k):
    return weibull_pdf(x, lam, k) / weibull_sf(x, lam, k)


# =========================
# 2. TYPE II HALF-LOGISTIC GENERATOR
# =========================

def tiihl_cdf(G, alpha):
    return (G ** alpha) / (G ** alpha + (1 - G) ** alpha)

def tiihl_pdf(G, g, alpha):
    num = alpha * G ** (alpha - 1) * (1 - G) ** (alpha - 1)
    den = (G ** alpha + (1 - G) ** alpha) ** 2
    return num / den * g


# =========================
# 3. SINE–TYPE II HALF-LOGISTIC WEIBULL
# =========================

def stiiHLW_cdf(x, lam, k, alpha):
    G = weibull_cdf(x, lam, k)
    H = tiihl_cdf(G, alpha)
    return np.sin(np.pi / 2 * H)

def stiiHLW_pdf(x, lam, k, alpha):
    G = weibull_cdf(x, lam, k)
    g = weibull_pdf(x, lam, k)
    H = tiihl_cdf(G, alpha)
    h_prime = tiihl_pdf(G, g, alpha)
    return (np.pi / 2) * np.cos(np.pi / 2 * H) * h_prime

def stiiHLW_sf(x, lam, k, alpha):
    return 1 - stiiHLW_cdf(x, lam, k, alpha)

def stiiHLW_hazard(x, lam, k, alpha):
    return stiiHLW_pdf(x, lam, k, alpha) / stiiHLW_sf(x, lam, k, alpha)
