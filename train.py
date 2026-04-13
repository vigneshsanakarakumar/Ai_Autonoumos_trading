"""
train.py
End-to-end training script: forecast model + DQN agent.

Usage:
    python train.py --ticker RELIANCE.NS --start 2018-01-01 --end 2023-12-31 \
                    --model transformer --dqn_episodes 300
"""

import argparse
import os
import numpy as np
import torch

from src.data.fetch_data import fetch_yahoo
from src.data.preprocess import prepare_dataset, add_technical_indicators
from src.models.lstm_model import LSTMForecaster, train_lstm
from src.models.transformer_model import TransformerForecaster, train_transformer
from src.rl_agent.environment import TradingEnv
from src.rl_agent.dqn_agent import DQNAgent
from src.utils.metrics import summarize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",       default="RELIANCE.NS")
    p.add_argument("--start",        default="2018-01-01")
    p.add_argument("--end",          default="2023-12-31")
    p.add_argument("--model",        default="transformer", choices=["lstm", "transformer"])
    p.add_argument("--seq_len",      type=int, default=60)
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--dqn_episodes", type=int, default=200)
    p.add_argument("--save_dir",     default="saved_models")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n=== Fetching data for {args.ticker} ===")
    df = fetch_yahoo(args.ticker, args.start, args.end)
    print(f"    {len(df)} trading days loaded.\n")

    # ── Forecasting model ──────────────────────────────────────────────
    print(f"=== Training {args.model.upper()} forecaster ===")
    X_tr, y_tr, X_te, y_te, scaler = prepare_dataset(df, seq_len=args.seq_len)
    n_feat = X_tr.shape[2]

    if args.model == "lstm":
        model = LSTMForecaster(input_size=n_feat)
        model = train_lstm(model, X_tr, y_tr, epochs=args.epochs)
    else:
        model = TransformerForecaster(input_size=n_feat)
        model = train_transformer(model, X_tr, y_tr, epochs=args.epochs)

    torch.save(model.state_dict(), f"{args.save_dir}/{args.model}_forecaster.pt")
    print(f"    Forecaster saved → {args.save_dir}/{args.model}_forecaster.pt\n")

    # ── DQN Agent ─────────────────────────────────────────────────────
    print("=== Training DQN Agent ===")
    df_feat = add_technical_indicators(df.copy())
    from sklearn.preprocessing import MinMaxScaler
    feat_cols = [c for c in ["Close","EMA_12","MACD","RSI","BB_Width","Log_Return","Volume"]
                 if c in df_feat.columns]
    arr = MinMaxScaler().fit_transform(df_feat[feat_cols].values.astype(np.float32))

    env = TradingEnv(arr, window=30)
    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n)

    rewards_history = []
    for ep in range(1, args.dqn_episodes + 1):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
            if done:
                break
        rewards_history.append(total_reward)
        if ep % 50 == 0:
            pv = info["portfolio_value"]
            print(f"  Episode {ep}/{args.dqn_episodes}  "
                  f"Total Reward: {total_reward:.4f}  "
                  f"Portfolio: ₹{pv:,.2f}  ε: {agent.epsilon:.3f}")

    agent.save(f"{args.save_dir}/dqn_agent.pt")
    print(f"\n    DQN agent saved → {args.save_dir}/dqn_agent.pt")
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
