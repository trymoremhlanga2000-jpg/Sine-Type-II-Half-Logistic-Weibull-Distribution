
# Sine–Type II Half-Logistic Weibull Distribution (STIIHLW)

## Overview

This repository presents an interactive, research-grade implementation of the Sine–Type II Half-Logistic Weibull (STIIHLW) distribution, a novel probability distribution constructed through a two-stage transformation of the Weibull distribution. The project integrates rigorous statistical theory with a modern Streamlit-based visualization framework to facilitate intuitive exploration of distributional behavior.

The application enables real-time investigation of the probability density function (PDF), cumulative distribution function (CDF), survival function, and hazard rate function, with full control over model parameters.


## Theoretical Background

The STIIHLW distribution is constructed as follows:

1. **Base Distribution**  
   The Weibull distribution serves as the foundational model, widely used in reliability engineering, survival analysis, and lifetime modeling.

2. **Type II Half-Logistic Generator**  
   Applied at the CDF level to introduce additional shape flexibility and tail reweighting while preserving the support.

3. **Sine Generator**  
   A trigonometric transformation applied to the transformed CDF, enhancing curvature, smoothness, and modal flexibility.

This layered construction yields a highly flexible distribution capable of modeling complex lifetime behaviors, including non-monotonic hazard functions.


## Features

- Interactive visualization of:
  - Probability Density Function (PDF)
  - Cumulative Distribution Function (CDF)
  - Survival Function
  - Hazard Function
- Real-time parameter control:
  - Scale parameter (λ)
  - Weibull shape parameter (k)
  - Type II Half-Logistic shape parameter (α)
- Side-by-side comparison between:
  - Base Weibull distribution
  - Sine–Type II Half-Logistic Weibull distribution


