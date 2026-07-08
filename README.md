# Options & Portfolio Quant Toolkit

A two-module quantitative finance toolkit built in Python — covering portfolio risk simulation and options pricing with full Greek derivation.

---

## Module 1 — Monte Carlo Portfolio Simulator

Simulates multi-asset portfolio behaviour over a full trading year using Monte Carlo methods.

### What It Does
- Simulates 100 random portfolio paths over 252 trading days
- Uses **Cholesky decomposition** to model realistic asset correlations
- Computes **VaR** and **CVaR** for tail risk quantification
- Plots the **Efficient Frontier** to identify optimal portfolio weights
- Calculates **Sharpe Ratio** for risk-adjusted performance measurement
- Exports a professional PDF report with embedded charts

### Results
- Best Sharpe Ratio: **3.08** across simulations
- Optimal weights led by AAPL and TCS.NS
- Efficient frontier plotted with min-variance and max-Sharpe portfolios highlighted

---

## Module 2 — Black-Scholes Options Pricing Engine

Quantitative pricing model for AAPL options using the Black-Scholes framework.

### What It Does
- Prices both **Call and Put options** using closed-form Black-Scholes formula
- Derives all five primary **Greeks**: Delta, Gamma, Vega, Theta, Rho
- Cross-validates pricing accuracy via **Monte Carlo simulation**
- Confirms model consistency between analytical and simulation-based results

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core logic |
| NumPy | Matrix ops, Cholesky decomposition |
| SciPy | Normal distribution functions for B-S |
| Pandas | Data handling |
| Matplotlib | Visualizations, frontier & payoff plots |
| yfinance | Historical price & volatility data |
| ReportLab | PDF report generation |

## Key Concepts

- Monte Carlo simulation for probabilistic portfolio modelling
- Cholesky decomposition for correlated asset returns
- Black-Scholes closed-form option valuation
- Greeks as sensitivity measures for risk management
- VaR / CVaR for tail risk under simulation

## Project Structure

```
options-and-portfolio-quant-toolkit/
│
├── mc.py               # Monte Carlo portfolio simulator
├── bs.py               # Black-Scholes pricing engine
├── report/             # Generated PDF outputs
├── charts/             # Exported visualizations
└── README.md
```

## Usage

```python
# Install dependencies
pip install numpy scipy pandas matplotlib yfinance reportlab

# Run portfolio simulator
python mc.py

# Run options pricing engine
python bs.py
```

---

*Part of my quantitative finance project series. Built independently to apply derivatives pricing theory and portfolio risk modelling in Python.*
