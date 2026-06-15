# ml_project
Financial Market Prediction &amp; Trading Strategy

Portfolio Optimization & ML Forecasting
This project analyzes a selection of ten major stock tickers using Modern Portfolio Theory (MPT) to optimize portfolio weights. Additionally, it integrates machine learning to forecast future stock prices and calculates potential downside risk for a recurring monthly investment strategy



Core Features
Automated Data Acquisition: Retrieves historical adjusted close prices from January 2021 to April 2026 using the yfinance API.  

Sharpe Ratio Maximization: Utilizes the pypfopt library to calculate the Efficient Frontier by evaluating historical returns and sample covariance.  

Actionable Allocation: Outputs precise dollar amounts for a $500 monthly contribution across the optimized assets.  

Time-Series Forecasting: Deploys the prophet machine learning model to generate 12-month price predictions for tech, healthcare, and defensive benchmark stocks.  

Value at Risk (VaR): Estimates the 95% monthly VaR to quantify the potential loss of the monthly contribution during high-volatility periods.  
