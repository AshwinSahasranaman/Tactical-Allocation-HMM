import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# DATA ACQUISITION & PREPROCESSING

def get_data():
    tickers = ["SPY", "TLT"]
    df = yf.download(tickers, start="2016-01-01", end="2026-01-01", progress=False)['Close']
    returns = np.log(df / df.shift(1)).dropna()
    return df, returns

# HMM MODEL TRAINING AND BACKTESTING

def run_hmm(returns):
    data = returns[['SPY']].copy()
    data['Vol'] = data['SPY'].rolling(20).std()
    data = data.dropna()
    
    train_data = data.loc[:"2023-12-31"]
    test_data = data.loc["2024-01-01":"2026-01-01"]

    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(train_data)
    
    regimes = model.predict(test_data)
    
    results = returns.loc[test_data.index].copy()
    results['Regime'] = regimes

    if train_data.groupby(model.predict(train_data))['SPY'].std().idxmax() == 0:
        results['Regime'] = 1 - results['Regime']
        
    return results, model

# RISK & PERFORMANCE METRICS

def calculate_metrics(df):
    def max_dd(rets):
        cum = np.exp(rets.cumsum()) 
        peak = cum.expanding(min_periods=1).max()
        return ((cum - peak) / peak).min()

    def cvar_95(rets):
        var_95 = np.percentile(rets, 5)
        avg_log_loss = rets[rets <= var_95].mean()
        return np.exp(avg_log_loss) - 1

    ann_ret = np.exp(df.mean() * 252) - 1
    ann_vol = df.std() * np.sqrt(252)

    summary = {
        "Annual Return (%)": ann_ret*100,
        "Annual Vol (%)": ann_vol*100,
        "Sharpe Ratio": ann_ret / ann_vol,
        "Max Drawdown (%)": max_dd(df)*100,
        "CVaR 95% (%)": cvar_95(df)*100
    }
    return pd.Series(summary)

# MAIN 

prices, returns = get_data()
results, hmm_model = run_hmm(returns)

trans_mat = pd.DataFrame(hmm_model.transmat_, 
                         columns=['To Low Vol', 'To High Vol'], 
                         index=['From Low Vol', 'From High Vol'])

results['Strategy_Ret'] = np.where(results['Regime'].shift(1) == 0, 
                                   results['SPY'], 
                                   results['TLT'])
results = results.dropna()

bench_perf = calculate_metrics(results['SPY'])
strat_perf = calculate_metrics(results['Strategy_Ret'])
comparison = pd.DataFrame({"Benchmark (SPY)": bench_perf, "Tactical Strategy": strat_perf})

# VISUALIZATION DASHBOARD

fig, axes = plt.subplots(3, 1, figsize=(14, 18))

# Plot Regimes

axes[0].plot(prices.loc[results.index].index, prices.loc[results.index, 'SPY'], color='black', alpha=0.3, label='SPY Price')
axes[0].scatter(results.index, prices.loc[results.index, 'SPY'], 
               c=results['Regime'], cmap='RdYlGn_r', s=10, label='Regime (Green=Low Volatility, Red=High Volatility)')
axes[0].set_title("Out-of-Sample Market Regime (2024-2026)", fontsize=16, fontweight='bold')
axes[0].set_ylabel("Close Price (USD)")
axes[0].legend()

# Plot Cumulative Returns 

strat_cum = np.exp(results['Strategy_Ret'].cumsum())
bench_cum = np.exp(results['SPY'].cumsum())

axes[1].plot(bench_cum, label='Benchmark (SPY)', color='gray', linestyle='--')
axes[1].plot(strat_cum, label='Regime-Switching Strategy', color='green', linewidth=2)
axes[1].set_title("Out-of-Sample Strategy Equity Curve (2024-2026)", fontsize=16, fontweight='bold')
axes[1].set_ylabel("Growth of $1 Investment")
axes[1].legend()
axes[1].set_ylim(0, 2) 

# Plot Correlation

rolling_corr = results['SPY'].rolling(60).corr(results['TLT'])
axes[2].plot(rolling_corr, color='blue', label='60D Rolling Corr')
axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[2].fill_between(rolling_corr.index, rolling_corr, where=(rolling_corr > 0), 
                 color='red', alpha=0.2, label='Positive Corr (Hedge Failure)')
axes[2].set_title("SPY - TLT Correlation Monitor (2024-2026)", fontsize=16, fontweight='bold')
axes[2].set_ylabel("Correlation")
axes[2].set_ylim(-1, 1) 

plt.tight_layout()
plt.show();

print("\nSTATE TRANSITION MATRIX")
print(trans_mat.round(4))
print("\nFINAL PERFORMANCE COMPARISON ON OUT OF SAMPLE DATA (2024-2026)")
print(comparison.round(2))