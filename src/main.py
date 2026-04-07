import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# DATA ACQUISITION & PREPROCESSING

def get_data():
    tickers = ["SPY"]
    df = yf.download(
        tickers,
        start="2010-01-01",
        end="2026-01-01",
        progress=False
    )["Close"]

    returns = np.log(df / df.shift(1)).dropna()
    return df, returns


# HMM MODEL TRAINING AND BACKTESTING

def run_hmm(returns):
    data = returns[["SPY"]].copy()
    data["Vol"] = data["SPY"].rolling(20).std()
    data = data.dropna()

    train_data = data.loc[:"2024-12-31"]
    test_data = data.loc["2025-01-01":"2026-01-01"]

    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    model.fit(train_data)

    train_states = model.predict(train_data)

    state_order = train_data.groupby(train_states)["SPY"].std().sort_values().index.tolist()
    state_map = {state: idx for idx, state in enumerate(state_order)}

    full_data = pd.concat([train_data, test_data])
    regimes = []
    train_len = len(train_data)

    for offset in range(len(test_data)):
        observed = full_data.iloc[:train_len + offset]
        inferred_states = model.predict(observed)
        regimes.append(state_map[inferred_states[-1]])

    results = returns.loc[test_data.index].copy()
    results["Regime"] = regimes

    reordered_transmat = model.transmat_[state_order][:, state_order]

    return results, model, reordered_transmat


# RISK & PERFORMANCE METRICS

def calculate_metrics(rets):
    def max_dd(log_rets):
        cum = np.exp(log_rets.cumsum())
        peak = cum.expanding(min_periods=1).max()
        return ((cum - peak) / peak).min()

    def cvar_95(log_rets):
        var_95 = np.percentile(log_rets, 5)
        avg_log_loss = log_rets[log_rets <= var_95].mean()
        return np.exp(avg_log_loss) - 1

    ann_ret = np.exp(rets.mean() * 252) - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = rets.mean() / rets.std() * np.sqrt(252)

    summary = {
        "Annual Return (%)": ann_ret * 100,
        "Annual Vol (%)": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd(rets) * 100,
        "CVaR 95% (%)": cvar_95(rets) * 100
    }
    return pd.Series(summary)


# MAIN

prices, returns = get_data()
results, hmm_model, transmat = run_hmm(returns)

trans_mat = pd.DataFrame(
    transmat,
    columns=["To Low Vol", "To High Vol"],
    index=["From Low Vol", "From High Vol"]
)

results["Equity_Weight"] = np.where(results["Regime"].shift(1) == 0, 1.0, 0.4)
results["Cash_Weight"] = 1.0 - results["Equity_Weight"]

results["Strategy_Ret"] = results["Equity_Weight"] * results["SPY"]

results = results.dropna()

bench_perf = calculate_metrics(results["SPY"])
strat_perf = calculate_metrics(results["Strategy_Ret"])
comparison = pd.DataFrame({
    "Benchmark (SPY)": bench_perf,
    "HMM SPY/Cash Strategy": strat_perf
})

# VISUALIZATION DASHBOARD

fig, axes = plt.subplots(2, 1, figsize=(14, 18))

# Plot Regimes
axes[0].plot(
    prices.loc[results.index].index,
    prices.loc[results.index, "SPY"],
    color="black",
    alpha=0.3,
    label="SPY Price"
)

axes[0].scatter(
    results.index,
    prices.loc[results.index, "SPY"],
    c=results["Regime"],
    cmap="RdYlGn_r",
    s=10,
    label="Regime (Green=Low Volatility, Red=High Volatility)"
)

axes[0].set_title("Market Regime (2025-2026)", fontsize=16, fontweight="bold")
axes[0].set_ylabel("Close Price (USD)")
axes[0].legend()

# Plot Cumulative Returns
strat_cum = np.exp(results["Strategy_Ret"].cumsum())
bench_cum = np.exp(results["SPY"].cumsum())

axes[1].plot(bench_cum, label="Benchmark (SPY)", color="gray", linestyle="--")
axes[1].plot(strat_cum, label="HMM SPY/Cash Strategy", color="green", linewidth=2)
axes[1].set_title("Strategy Equity Curve (2025-2026)", fontsize=16, fontweight="bold")
axes[1].set_ylabel("Growth of $1 Investment")
axes[1].legend()


plt.tight_layout()
plt.savefig("regime_detection_dashboard.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSTATE TRANSITION MATRIX")
print(trans_mat.round(4))

print("\nFINAL PERFORMANCE COMPARISON ON OUT-OF-SAMPLE DATA (2025-2026)")
print(comparison.round(2))