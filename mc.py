import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# Import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change(fill_method=None)  # fixes FutureWarning
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# Global stock list
usa_stocks    = ['JPM', 'GS', 'AAPL', 'MSFT', 'XOM']
india_stocks  = ['HDFCBANK.NS', 'TCS.NS', 'RELIANCE.NS'] 
europe_stocks = ['HSBA.L', 'BP.L']

stocks = usa_stocks + india_stocks + europe_stocks

# Date range
start = dt.datetime(2020, 1, 1)
end   = dt.datetime.now()

# Get data
meanReturns, covMatrix = get_data(stocks, start, end)

# Monte Carlo Settings
mc_sims   = 100
T         = 252
numStocks = len(stocks)

meanM = np.full(shape=(T, numStocks), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)  # fixed size

initialPortfolio = 10000

for m in range(mc_sims):  # this goes 0 to 99, matching mc_sims=100
    weights = np.random.random(numStocks)
    weights /= np.sum(weights)

    Z = np.random.normal(size=(T, numStocks))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)

    portfolio_sims[:, m] = np.cumprod(
        np.inner(weights, dailyReturns.T) + 1
    ) * initialPortfolio

# Plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# Import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change(fill_method=None)  # fixes FutureWarning
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# Global stock list
usa_stocks    = ['JPM', 'GS', 'AAPL', 'MSFT', 'XOM']
india_stocks  = ['HDFCBANK.NS', 'TCS.NS', 'RELIANCE.NS']
europe_stocks = ['HSBA.L', 'BP.L']

stocks = usa_stocks + india_stocks + europe_stocks

# Date range
start = dt.datetime(2020, 1, 1)
end   = dt.datetime.now()

# Get data
meanReturns, covMatrix = get_data(stocks, start, end)

# Monte Carlo Settings
mc_sims   = 100
T         = 252
numStocks = len(stocks)

meanM = np.full(shape=(T, numStocks), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)  # fixed size

initialPortfolio = 10000

for m in range(mc_sims):  # this goes 0 to 99, matching mc_sims=100
    weights = np.random.random(numStocks)
    weights /= np.sum(weights)

    Z = np.random.normal(size=(T, numStocks))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)

    portfolio_sims[:, m] = np.cumprod(
        np.inner(weights, dailyReturns.T) + 1
    ) * initialPortfolio

# Plot
# Confidence Interval Bands
percentile_5  = np.percentile(portfolio_sims, 5,  axis=1)
percentile_50 = np.percentile(portfolio_sims, 50, axis=1)
percentile_95 = np.percentile(portfolio_sims, 95, axis=1)

# Plot
plt.style.use('dark_background')
plt.figure(figsize=(12, 6))

# All simulation lines (faint)
plt.plot(portfolio_sims, alpha=0.1, linewidth=0.5, color='cyan')

# Confidence bands
plt.plot(percentile_95, color='green',  linewidth=2, label='95th Percentile (Best)')
plt.plot(percentile_50, color='yellow', linewidth=2, label='50th Percentile (Median)')
plt.plot(percentile_5,  color='red',    linewidth=2, label='5th Percentile (Worst)')

# Shaded area between bands
plt.fill_between(range(T), percentile_5, percentile_95, alpha=0.15, color='cyan', label='90% Confidence Zone')

# Initial investment line
plt.axhline(y=initialPortfolio, color='white', linestyle='--', linewidth=1.5, label='Initial Investment')

plt.title('Monte Carlo Simulation - Global Portfolio', fontsize=14, color='white')
plt.xlabel('Trading Days', color='white')
plt.ylabel('Portfolio Value (USD)', color='white')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Results
best_case  = np.max(portfolio_sims[-1])
worst_case = np.min(portfolio_sims[-1])
avg_case   = np.mean(portfolio_sims[-1])

print(f"\nInitial Investment : ${initialPortfolio:,.2f}")
print(f"Best Case  (1 year): ${best_case:,.2f}")
print(f"Worst Case (1 year): ${worst_case:,.2f}")
print(f"Average    (1 year): ${avg_case:,.2f}")

# Results
best_case  = np.max(portfolio_sims[-1])
worst_case = np.min(portfolio_sims[-1])
avg_case   = np.mean(portfolio_sims[-1])

print(f"\nInitial Investment : ${initialPortfolio:,.2f}")
print(f"Best Case  (1 year): ${best_case:,.2f}")
print(f"Worst Case (1 year): ${worst_case:,.2f}")
print(f"Average    (1 year): ${avg_case:,.2f}")

#Var Calculation
confidenceLevel = 0.05 # 5% = 95% confidence 

#sort final portfolio values and find the cutoff
finalValues = portfolio_sims[-1]
VaR         = initialPortfolio - np.percentile(finalValues, confidenceLevel *100)
CVaR        = initialPortfolio - np.mean(finalValues[finalValues <=np.percentile(finalValues, confidenceLevel * 100)])

print(f"\n--- Risk Metrices ---")
print(f"VaR  (95% confidence): You could lose up to ${VaR:,.2f}")
print(f"CVaR (95% confidence): In the worst 5% cases, average loss is ${CVaR:,.2f}")

# Sharpe Ratio
riskFreeRate = 0.04  # 4% US Treasury rate

# Calculate for each simulation
returns_sim  = (portfolio_sims[-1] - initialPortfolio) / initialPortfolio
sharpe_ratios = (returns_sim - riskFreeRate) / returns_sim.std()

best_sharpe  = np.max(sharpe_ratios)
worst_sharpe = np.min(sharpe_ratios)
avg_sharpe   = np.mean(sharpe_ratios)

print("="*40)
print("--- Sharpe Ratios ---")
print(f"Best  Simulation: {best_sharpe:.2f}")
print(f"Worst Simulation: {worst_sharpe:.2f}")
print(f"Average         : {avg_sharpe:.2f}")
print("="*40)

# Optimal Weights
# Run simulations storing weights and sharpe ratios
mc_sims_opt = 1000  # more simulations = better optimization

results     = np.zeros((3, mc_sims_opt))  # return, risk, sharpe
weights_store = []

for m in range(mc_sims_opt):
    # Random weights
    weights = np.random.random(numStocks)
    weights /= np.sum(weights)
    weights_store.append(weights)

    # Annualized return and risk
    port_return = np.sum(meanReturns * weights) * 252
    port_risk   = np.sqrt(
        np.dot(weights.T, np.dot(covMatrix * 252, weights))
    )

    results[0, m] = port_return
    results[1, m] = port_risk
    results[2, m] = (port_return - riskFreeRate) / port_risk  # Sharpe

# Find best Sharpe Ratio
best_idx     = np.argmax(results[2])
best_weights = weights_store[best_idx]

print("="*40)
print("--- Optimal Portfolio Weights ---")
for i, stock in enumerate(stocks):
    print(f"{stock:15s} : {best_weights[i]*100:.2f}%")
print(f"\nExpected Annual Return : {results[0, best_idx]*100:.2f}%")
print(f"Expected Annual Risk   : {results[1, best_idx]*100:.2f}%")
print(f"Best Sharpe Ratio      : {results[2, best_idx]:.2f}")
print("="*40)

# Plot efficient frontier
plt.style.use('dark_background')
plt.figure(figsize=(12, 6))
plt.scatter(
    results[1, :],
    results[0, :],
    c=results[2, :],       # color by sharpe ratio
    cmap='viridis',
    alpha=0.5,
    s=10
)
plt.colorbar(label='Sharpe Ratio')

# Highlight optimal portfolio
plt.scatter(
    results[1, best_idx],
    results[0, best_idx],
    color='red',
    marker='*',
    s=300,
    label='Optimal Portfolio'
)

plt.title('Efficient Frontier - Global Portfolio', fontsize=14)
plt.xlabel('Annual Risk (Volatility)')
plt.ylabel('Annual Return')
plt.legend()
plt.tight_layout()
plt.show()