# Autonomous AI Financial Trading Agent

> **23CSR405 – ADVANCED ALGORITHMS | ASSIGNMENT II (MINI PROJECT)**  
> Department of Computer Science and Engineering  
> Academic Year: 2025–26

---

## Overview

An Autonomous AI Trading Agent that predicts stock trends using LSTM/Transformer models, optimizes buy/sell decisions with Deep Q-Network (DQN) reinforcement learning, and applies Markowitz portfolio optimization — all powered by real market data from Yahoo Finance and NSE India.

---

## Features

- **Time-Series Forecasting** – LSTM and Transformer models for stock price prediction
- **Reinforcement Learning** – Deep Q-Network (DQN) for autonomous buy/sell/hold decisions
- **Portfolio Optimization** – Markowitz Mean-Variance Optimization
- **Priority Queue Asset Tracker** – Max-heap to rank top-performing assets in real time
- **Risk Management** – Sharpe Ratio, VaR, dynamic rebalancing
- **Interactive Dashboard** – Streamlit UI for live signals, portfolio view, and performance charts

---

## Project Structure

```
autonomous-ai-trading-agent/
├── src/
│   ├── data/
│   │   ├── fetch_data.py          # Yahoo Finance & NSE data fetcher
│   │   └── preprocess.py          # Feature engineering & normalization
│   ├── models/
│   │   ├── lstm_model.py          # LSTM forecasting model
│   │   └── transformer_model.py   # Transformer forecasting model
│   ├── rl_agent/
│   │   ├── dqn_agent.py           # Deep Q-Network agent
│   │   ├── environment.py         # Custom trading gym environment
│   │   └── replay_buffer.py       # Experience replay memory
│   ├── portfolio/
│   │   ├── markowitz.py           # Mean-variance optimization
│   │   └── asset_tracker.py       # Priority queue / max-heap tracker
│   └── utils/
│       ├── metrics.py             # Sharpe ratio, drawdown, VaR
│       └── visualize.py           # Chart utilities
├── app.py                         # Streamlit dashboard
├── train.py                       # Training pipeline
├── evaluate.py                    # Backtesting & evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Forecasting | LSTM, Transformer (PyTorch) |
| RL Agent | Deep Q-Network (DQN) |
| Optimization | Markowitz (scipy, cvxpy) |
| Data | yfinance, NSEpy |
| Dashboard | Streamlit |
| Tracking | heapq (Priority Queue) |

---

## Setup

```bash
git clone https://github.com/yourusername/autonomous-ai-trading-agent.git
cd autonomous-ai-trading-agent
pip install -r requirements.txt
```

### Train models
```bash
python train.py --ticker RELIANCE.NS --epochs 50
```

### Run backtesting
```bash
python evaluate.py --ticker TCS.NS --start 2020-01-01 --end 2024-01-01
```

### Launch Streamlit dashboard
```bash
streamlit run app.py
```

---

## Datasets

- **Yahoo Finance** – Global equities via `yfinance`
- **NSE India** – Indian market data via `NSEpy` or direct CSV

---

## Results (Sample)

| Model | MAE | RMSE | Sharpe Ratio |
|---|---|---|---|
| LSTM | 4.2 | 6.1 | 1.43 |
| Transformer | 3.8 | 5.5 | 1.61 |
| DQN Agent | – | – | 1.87 |

---

## License

MIT License
