# Tactical Asset Allocation & Risk Mitigation Strategy Using HMM

This project implements a **Gaussian Hidden Markov Model (HMM)** to identify latent market regimes—specifically "Low Volatility" and "High Volatility" states. By detecting these shifts, the strategy tactically rotates between Equities (SPY) and Bonds (TLT) to optimize risk-adjusted returns.

## Performance Analysis
The HMM was trained on data from 2016-01-01 to 2023-12-31. The strategy was backtested using data from 2024-01-01 to 2026-01-01. The results demonstrate a significant improvement in portfolio efficiency through regime-based switching.

| Metric | Benchmark (SPY) | Tactical Strategy | Improvement |
| :--- | :--- | :--- | :--- |
| **Annual Return** | 21.34% | 19.12% | -10.40% |
| **Annual Vol** | 16.28% | **12.41%** | **-23.77%** |
| **Sharpe Ratio** | 1.31 | **1.54** | **+17.55%** |
| **Max Drawdown** | -18.76% | -12.16% | **-35.18%** |
| **CVaR (95%)** | -2.35% | **-1.84%** | **-21.70%** |

![Market Regimes](regime_detection_dashboard.png) 

### **Key Results**
* **Risk-Adjusted Return Trade-off:** The strategy outperformed the benchmark for majority of the backtesting period. While the terminal annual return trailed the benchmark by 10.40% on a relative basis (2.22% absolute), this trade off resulted in a reduction of the risk exposure as shown by the metrics below.
* **Increase in Sharpe Ratio:** The strategy delivered a Sharpe Ratio of 1.54 representing a 17.55% improvement over the benchmark. This was achieved by mitigating equity exposure during high-volatility periods by switching from SPY to TLT.
* **Tail Risk Mitigation:** The Conditional Value at Risk (CVaR) improved by 0.51%, proving the HMM's ability to exit the market before the left-tail events materialized.

>>>>>>> 8738295 (Initial Commit: Tactical HMM Project)
