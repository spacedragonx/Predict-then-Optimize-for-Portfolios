# ============================
# SPO LSTM Training Script
# ============================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from models import*

# ----------------------------
# 1. Device (CPU / GPU)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Load processed data safely
#    (PyTorch 2.6+ best practice)
# ----------------------------
data = torch.load("data_save/25_spo_data.pt", weights_only=True)

X = data["X"]          # (time, features_total)
y = data["y"]          # (time, assets)
dates = data["dates"]
tickers = data["tickers"]

num_assets = len(tickers)
num_features = X.shape[1] // num_assets



# ----------------------------
# 3. Create dataset FIRST,
#    then split (prevents leakage)
# ----------------------------
SEQ_LEN = 20
dataset = FinancialDataset(X, y, num_assets, seq_len=SEQ_LEN)

split_idx = int(0.8 * len(dataset))

train_ds = torch.utils.data.Subset(dataset, range(0, split_idx))
test_ds  = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)


model = PortfolioNet(
    num_assets=num_assets,
    num_features=num_features
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# 4. Training Loop
# ----------------------------
epochs = 25
train_losses = []

print("\nStarting Training...\n")
all_weights = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for X_seq, y_next in train_loader:
        X_seq = X_seq.to(device)
        y_next = y_next.to(device)

        optimizer.zero_grad()

        weights = model(X_seq)
        
        loss = spo_sharpe_loss(weights, y_next)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Negative Sharpe: {avg_loss:.4f}")

# ----------------------------
# 5. Evaluation / Backtest
# ----------------------------
model.eval()

portfolio_returns = []
benchmark_returns = []

with torch.no_grad():
    for X_seq, y_next in test_loader:
        X_seq = X_seq.to(device)
        y_next = y_next.to(device)

        weights = model(X_seq)
        all_weights.append(weights.cpu())
        # Portfolio daily return
        port_ret = torch.sum(weights * y_next, dim=1)
        portfolio_returns.append(port_ret.cpu())

        # Equal-weight benchmark
        bench_ret = torch.mean(y_next, dim=1)
        benchmark_returns.append(bench_ret.cpu())

portfolio_returns = torch.cat(portfolio_returns).numpy()
benchmark_returns = torch.cat(benchmark_returns).numpy()
all_weights = torch.cat(all_weights).numpy()

equity_curve = np.cumprod(1 + portfolio_returns)
bench_curve  = np.cumprod(1 + benchmark_returns)

# ----------------------------
# 6. Saving the Model and Results for plotting
# ----------------------------
results = {
    "portfolio_returns": portfolio_returns,
    "benchmark_returns": benchmark_returns,
    "weights": all_weights,
    "tickers": tickers
}

torch.save(results, "Results/test_outputs.pt")
checkpoint = {
    "model_state_dict": model.state_dict(),
    "num_assets": len(tickers),
    "optimizer_state_dict": optimizer.state_dict(),
    "num_features": X.shape[1] // len(tickers),
    "seq_len": 20,
    "tickers": tickers
}

torch.save(checkpoint, "Trained_Models/spo_portfolio_model2.pt")
print("Model saved in Trained_Models Folder")
# ----------------------------
# 7. Plot results
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(equity_curve, label="SPO LSTM Strategy", linewidth=2)
plt.plot(bench_curve, label="Equal Weight Benchmark", linestyle="--")
plt.title("SPO LSTM Portfolio vs Benchmark")
plt.xlabel("Trading Days")
plt.ylabel("Normalized Portfolio Value")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("Graphs/spo_vs_benchmark.png", dpi=300, bbox_inches="tight")
plt.show()