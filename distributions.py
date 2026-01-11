import numpy as np
from scipy.special import gamma, gammainc, digamma
from scipy.optimize import minimize
import warnings

def weibull_pdf(x, lam, k):
    """Weibull probability density function"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (k/lam) * (x/lam)**(k-1) * np.exp(-(x/lam)**k)

def weibull_cdf(x, lam, k):
    """Weibull cumulative distribution function"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1 - np.exp(-(x/lam)**k)

def weibull_sf(x, lam, k):
    """Weibull survival function"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.exp(-(x/lam)**k)

def weibull_hazard(x, lam, k):
    """Weibull hazard function"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf = weibull_pdf(x, lam, k)
        sf = weibull_sf(x, lam, k)
        return np.where(sf > 0, pdf/sf, 0)

def stiiHLW_pdf(x, lam, k, alpha):
    """STIIHL Weibull probability density function"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = weibull_cdf(x, lam, k)
        g = weibull_pdf(x, lam, k)
        
        # Handle edge cases
        G = np.clip(G, 1e-15, 1-1e-15)
        
        T = G**alpha / (G**alpha + (1-G)**alpha)
        dT_dG = alpha * G**(alpha-1) * (1-G)**(alpha-1) / (G**alpha + (1-G)**alpha)**2
        
        return (np.pi/2) * np.cos((np.pi/2) * T) * dT_dG * g

def stiiHLW_cdf(x, lam, k, alpha):
    """STIIHL Weibull cumulative distribution function"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = weibull_cdf(x, lam, k)
        
        # Handle edge cases
        G = np.clip(G, 1e-15, 1-1e-15)
        
        T = G**alpha / (G**alpha + (1-G)**alpha)
        return np.sin((np.pi/2) * T)

def stiiHLW_sf(x, lam, k, alpha):
    """STIIHL Weibull survival function"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1 - stiiHLW_cdf(x, lam, k, alpha)

def stiiHLW_hazard(x, lam, k, alpha):
    """STIIHL Weibull hazard function"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf = stiiHLW_pdf(x, lam, k, alpha)
        sf = stiiHLW_sf(x, lam, k, alpha)
        return np.where(sf > 0, pdf/sf, 0)

def stiiHLW_quantile(p, lam, k, alpha):
    """STIIHL Weibull quantile function (inverse CDF)"""
    if p <= 0:
        return 0
    if p >= 1:
        return np.inf
    
    # Numerically solve for x such that CDF(x) = p
    from scipy.optimize import brentq
    
    def func(x):
        return stiiHLW_cdf(np.array([x]), lam, k, alpha)[0] - p
    
    try:
        # Find reasonable bounds
        max_iter = 100
        upper = lam * (-np.log(1e-10))**(1/k) * 10
        return brentq(func, 0, upper)
    except:
        return np.nan

def mle_stiiHLW(data):
    """Maximum Likelihood Estimation for STIIHL Weibull"""
    def neg_log_likelihood(params):
        lam, k, alpha = params
        if lam <= 0 or k <= 0 or alpha <= 0:
            return np.inf
        pdf_vals = stiiHLW_pdf(data, lam, k, alpha)
        pdf_vals = np.clip(pdf_vals, 1e-15, None)
        return -np.sum(np.log(pdf_vals))
    
    # Initial guesses based on data
    initial_lam = np.mean(data)
    initial_k = 2.0
    initial_alpha = 1.0
    
    bounds = [(0.1, 10*np.max(data)), (0.1, 10), (0.1, 10)]
    result = minimize(neg_log_likelihood, 
                     [initial_lam, initial_k, initial_alpha],
                     bounds=bounds,
                     method='L-BFGS-B')
    
    if result.success:
        return result.x
    else:
        return [initial_lam, initial_k, initial_alpha]

def goodness_of_fit(data, lam, k, alpha):
    """Calculate goodness of fit statistics"""
    n = len(data)
    
    # Kolmogorov-Smirnov statistic
    sorted_data = np.sort(data)
    ecdf = np.arange(1, n+1) / n
    tcdf = stiiHLW_cdf(sorted_data, lam, k, alpha)
    ks_stat = np.max(np.abs(ecdf - tcdf))
    
    # AIC and BIC
    log_lik = -np.sum(np.log(stiiHLW_pdf(data, lam, k, alpha)))
    aic = 2 * 3 + 2 * log_lik  # 3 parameters
    bic = 3 * np.log(n) + 2 * log_lik
    
    return {
        'KS Statistic': ks_stat,
        'AIC': aic,
        'BIC': bic,
        'Log-Likelihood': -log_lik
    }

def generate_stiiHLW_samples(n, lam, k, alpha):
    """Generate random samples from STIIHL Weibull distribution"""
    u = np.random.uniform(0, 1, n)
    samples = np.array([stiiHLW_quantile(p, lam, k, alpha) for p in u])
    return samples