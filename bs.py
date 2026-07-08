"""
Black-Scholes Options Pricing Model
Underlying Asset: AAPL
"""

import numpy as np
import yfinance as yf
import datetime as dt
from scipy.stats import norm
import matplotlib.pyplot as plt

# ================================
# Step 1: Fetch AAPL Live Data
# ================================
ticker = 'AAPL'
stock  = yf.Ticker(ticker)

# Get current price
currentPrice = stock.history(period='1d')['Close'].iloc[-1]
print(f"AAPL Current Price: ${currentPrice:.2f}")

# Calculate historical volatility (sigma)
historicalData = stock.history(period='1y')['Close']
logReturns     = np.log(historicalData / historicalData.shift(1))
sigma          = logReturns.std() * np.sqrt(252)  # annualized
print(f"AAPL Historical Volatility: {sigma*100:.2f}%")

# ================================
# Step 2: Define Option Parameters
# ================================
S = currentPrice   # current stock price
K = 220            # strike price
T = 0.25           # time to expiry (3 months)
r = 0.04           # risk free rate (4%)

# ================================
# Step 3: Black-Scholes Formula
# ================================
def black_scholes(S, K, T, r, sigma):
    # d1 and d2 are intermediate calculations
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Call and Put prices
    callPrice = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    putPrice  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return callPrice, putPrice

callPrice, putPrice = black_scholes(S, K, T, r, sigma)

print("\n" + "="*40)
print("--- Black-Scholes Results ---")
print(f"Stock Price    : ${S:.2f}")
print(f"Strike Price   : ${K:.2f}")
print(f"Expiry         : {T*12:.0f} months")
print(f"Volatility     : {sigma*100:.2f}%")
print(f"Risk Free Rate : {r*100:.0f}%")
print("="*40)
print(f"CALL Option Price : ${callPrice:.2f}")
print(f"PUT  Option Price : ${putPrice:.2f}")
print("="*40)

# ================================
# Step 4: The Greeks
# ================================
def calculate_greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    callDelta = norm.cdf(d1)
    putDelta  = -norm.cdf(-d1)

    # Gamma (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta
    callTheta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    putTheta  = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega (same for call and put)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Rho
    callRho = K * T * np.exp(-r * T) * norm.cdf(d2)  / 100
    putRho  = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return callDelta, putDelta, gamma, callTheta, putTheta, vega, callRho, putRho

callDelta, putDelta, gamma, callTheta, putTheta, vega, callRho, putRho = calculate_greeks(S, K, T, r, sigma)

print("\n--- The Greeks ---")
print(f"{'Greek':<12} {'Call':>10} {'Put':>10}")
print("-" * 35)
print(f"{'Delta':<12} {callDelta:>10.4f} {putDelta:>10.4f}")
print(f"{'Gamma':<12} {gamma:>10.4f} {gamma:>10.4f}")
print(f"{'Theta':<12} {callTheta:>10.4f} {putTheta:>10.4f}")
print(f"{'Vega':<12} {vega:>10.4f} {vega:>10.4f}")
print(f"{'Rho':<12} {callRho:>10.4f} {putRho:>10.4f}")
print("-" * 35)

# Plot Greeks vs Stock Price
stockPrices = np.linspace(150, 350, 200)
callDeltas  = [calculate_greeks(p, K, T, r, sigma)[0] for p in stockPrices]
putDeltas   = [calculate_greeks(p, K, T, r, sigma)[1] for p in stockPrices]
callThetas  = [calculate_greeks(p, K, T, r, sigma)[3] for p in stockPrices]
gammas      = [calculate_greeks(p, K, T, r, sigma)[2] for p in stockPrices]

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('AAPL Options - The Greeks', fontsize=16, color='white')

# Delta
axes[0,0].plot(stockPrices, callDeltas, color='green', linewidth=2, label='Call Delta')
axes[0,0].plot(stockPrices, putDeltas,  color='red',   linewidth=2, label='Put Delta')
axes[0,0].axvline(x=S, color='white', linestyle='--', linewidth=1, label=f'Current Price ${S:.0f}')
axes[0,0].set_title('Delta', color='white')
axes[0,0].legend()
axes[0,0].set_xlabel('Stock Price')
axes[0,0].set_ylabel('Delta')

# Gamma
axes[0,1].plot(stockPrices, gammas, color='cyan', linewidth=2)
axes[0,1].axvline(x=S, color='white', linestyle='--', linewidth=1, label=f'Current Price ${S:.0f}')
axes[0,1].set_title('Gamma', color='white')
axes[0,1].legend()
axes[0,1].set_xlabel('Stock Price')
axes[0,1].set_ylabel('Gamma')

# Theta
axes[1,0].plot(stockPrices, callThetas, color='yellow', linewidth=2, label='Call Theta')
axes[1,0].axvline(x=S, color='white', linestyle='--', linewidth=1, label=f'Current Price ${S:.0f}')
axes[1,0].set_title('Theta (Daily Decay)', color='white')
axes[1,0].legend()
axes[1,0].set_xlabel('Stock Price')
axes[1,0].set_ylabel('Theta')

# Vega
vegas = [calculate_greeks(p, K, T, r, sigma)[5] for p in stockPrices]
axes[1,1].plot(stockPrices, vegas, color='orange', linewidth=2)
axes[1,1].axvline(x=S, color='white', linestyle='--', linewidth=1, label=f'Current Price ${S:.0f}')
axes[1,1].set_title('Vega', color='white')
axes[1,1].legend()
axes[1,1].set_xlabel('Stock Price')
axes[1,1].set_ylabel('Vega')

plt.tight_layout()
plt.show()

# ================================
# Step 5: Monte Carlo Options Pricing
# ================================
mc_sims = 1000   # number of simulations
T_days  = 63     # 3 months = 63 trading days

print("\nRunning Monte Carlo Options Pricing...")

# Simulate AAPL price paths
np.random.seed(42)
dt_mc        = T / T_days
pricesPaths  = np.zeros((T_days, mc_sims))
pricesPaths[0] = S

for t in range(1, T_days):
    Z = np.random.standard_normal(mc_sims)
    pricesPaths[t] = pricesPaths[t-1] * np.exp(
        (r - 0.5 * sigma**2) * dt_mc + sigma * np.sqrt(dt_mc) * Z
    )

# Calculate payoffs at expiry
callPayoffs = np.maximum(pricesPaths[-1] - K, 0)  # max(S-K, 0)
putPayoffs  = np.maximum(K - pricesPaths[-1], 0)  # max(K-S, 0)

# Discount back to present value
mc_callPrice = np.exp(-r * T) * np.mean(callPayoffs)
mc_putPrice  = np.exp(-r * T) * np.mean(putPayoffs)

print("\n" + "="*40)
print("--- Monte Carlo vs Black-Scholes ---")
print(f"{'Method':<20} {'Call':>8} {'Put':>8}")
print("-"*40)
print(f"{'Black-Scholes':<20} ${callPrice:>7.2f} ${putPrice:>7.2f}")
print(f"{'Monte Carlo':<20} ${mc_callPrice:>7.2f} ${mc_putPrice:>7.2f}")
print(f"{'Difference':<20} ${abs(callPrice-mc_callPrice):>7.2f} ${abs(putPrice-mc_putPrice):>7.2f}")
print("="*40)

# Plot simulated price paths
plt.style.use('dark_background')
plt.figure(figsize=(12, 6))
plt.plot(pricesPaths[:, :50], alpha=0.2, linewidth=0.8)  # plot 50 paths
plt.axhline(y=K, color='red',   linestyle='--', linewidth=2, label=f'Strike Price ${K}')
plt.axhline(y=S, color='white', linestyle='--', linewidth=1, label=f'Current Price ${S:.0f}')
plt.title('AAPL Monte Carlo Price Simulation (1000 paths)', fontsize=14)
plt.xlabel('Trading Days')
plt.ylabel('AAPL Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot payoff distribution
plt.figure(figsize=(12, 5))
plt.hist(callPayoffs, bins=50, color='green', alpha=0.6, label='Call Payoffs')
plt.hist(putPayoffs,  bins=50, color='red',   alpha=0.6, label='Put Payoffs')
plt.axvline(x=np.mean(callPayoffs), color='yellow', linewidth=2, label=f'Avg Call Payoff ${np.mean(callPayoffs):.2f}')
plt.title('Option Payoff Distribution at Expiry', fontsize=14)
plt.xlabel('Payoff (USD)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()