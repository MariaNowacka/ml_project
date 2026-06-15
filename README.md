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


FOR Portfolio optimizer:

Key Features
Data Pipeline: Automated historical data acquisition via yfinance.

Markowitz Baseline: Calculates a static optimal portfolio using PyPortfolioOpt on training data to establish a rigorous performance benchmark.

Deep Learning Oracle: A robust Feedforward Neural Network (FNN) built with PyTorch, featuring Batch Normalization, LeakyReLU, and Dropout layers to combat financial noise and overfitting.

Active Trading Engine: Simulates daily portfolio rebalancing with smoothed execution (20% transition per day) and a defensive Cash-position logic when the network predicts negative returns across the board.

Quantitative Analytics: Automatically computes industry-standard metrics including CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.

Advanced Visualization: Generates a 4-panel Matplotlib dashboard comparing cumulative returns, drawdown profiles, return distributions, and dynamic capital allocation.

Requirements & Dependencies
Ensure you have Python 3.9+ installed. You can install the required libraries using pip:

pip install torch numpy pandas yfinance matplotlib seaborn scikit-learn pypfopt mlflow

At the very top of the script, you will find the Configuration & Dates section. You can manually adjust these parameters to backtest the strategy over different market regimes:

Python
# Set the start date for training the neural network
TRAIN_START = "2020-01-01" 

# Set the active trading simulation period (Out-of-Sample)
SIMULATION_START = "2024-06-01"
SIMULATION_END = "2026-04-27"

# Sequence length (days of history the NN looks at to predict T+1)
SEQ_LENGTH = 60 

# Your portfolio assets
TICKERS = ["UNH", "MSFT", "JPM", "LLY", "COST", "V", "WM", "NEE", "MSCI", "PG"]

Upon successful execution, the script will output two main artifacts:

Console Tearsheet: A formatted table comparing the ML Portfolio directly against the S&P 500 benchmark across key risk/reward metrics.

Analytics Dashboard: A pop-up Matplotlib window containing:

Cumulative Capital Growth: A line chart showing the growth of the ML portfolio vs. the Baseline and S&P 500.

Drawdown Profile: A visual representation of peak-to-trough declines.

Daily Returns Distribution: A KDE plot to analyze skewness and fat tails.

Capital Allocation History: A stacked area chart showing exactly which assets the algorithm bought and sold over time (including white space for uninvested cash).

🧪 Tracking with MLflow (Optional)
The architecture is designed to be easily integrated with MLflow. If you wish to track your FNN training runs, simply uncomment/add the mlflow.start_run() wrappers around the PyTorch training loop to automatically log parameters, MSE loss, and models to your local MLflow UI.
