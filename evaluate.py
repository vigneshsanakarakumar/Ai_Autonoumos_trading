"""
evaluate.py
Backtesting pipeline: loads a trained DQN agent, runs on test period,
and prints performance metrics + comparison with buy-and-hold.

Usage:
    python evaluate.py --ticker TCS.NS --start 2023-01-01 --end 2024-01-01
"""

import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.data.fetch_data import fetch_yahoo
from src.data.preprocess import add_technical_indicators
from src.rl_agent.environment import TradingEnv
from src.rl_agent.dqn_agent import DQNAgent
from src.utils.metrics import summarize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",    default="TCS.NS")
    p.add_argument("--start",     default="2023-01-01")
    p.add_argument("--end",       default="2024-01-01")
    p.add_argument("--model_dir", default="saved_models")
    p.add_argument("--capital",   type=float, default=100_000.0)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n=== Backtesting {args.ticker} [{args.start} → {args.end}] ===\n")

    df = fetch_yahoo(args.ticker, args.start, args.end)
    df_feat = add_technical_indicators(df.copy())

    feat_cols = [c for c in ["Close","EMA_12","MACD","RSI","BB_Width","Log_Return","Volume"]
                 if c in df_feat.columns]
    arr = MinMaxScaler().fit_transform(df_feat[feat_cols].values.astype(np.float32))

    env = TradingEnv(arr, window=30, initial_cash=args.capital)
    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n)
    agent.load(f"{args.model_dir}/dqn_agent.pt")
    agent.epsilon = 0.0   # greedy during evaluation

    state = env.reset()
    equity_curve = [args.capital]
    actions_taken = []

    while True:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        equity_curve.append(info["portfolio_value"])
        actions_taken.append(action)
        if done:
            break

    equity_curve = np.array(equity_curve)

    # Buy-and-hold baseline
    close_prices = df["Close"].values[-len(equity_curve):]
    bah = args.capital * close_prices / close_prices[0]

    print("── DQN Agent Performance ──────────────────")
    for k, v in summarize(equity_curve).items():
        print(f"  {k:<18}: {v}")

    print("\n── Buy & Hold Baseline ────────────────────")
    for k, v in summarize(bah).items():
        print(f"  {k:<18}: {v}")

    action_labels = {0: "Hold", 1: "Buy", 2: "Sell"}
    counts = {l: actions_taken.count(i) for i, l in action_labels.items()}
    print(f"\nAction breakdown: {counts}")


if __name__ == "__main__":
    main()
