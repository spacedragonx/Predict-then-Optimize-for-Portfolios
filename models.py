import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class FinancialDataset(Dataset):
    def __init__(self, X, y, num_assets, seq_len=20):
        """
        X: Tensor (time, features_total)
        y: Tensor (time, num_assets)
        """
        self.seq_len = seq_len
        self.num_assets = num_assets
        self.feats_per_asset = X.shape[1] // num_assets

        # (time, features_total) → (time, assets, features)
        self.X = X.reshape(-1, num_assets, self.feats_per_asset)
        self.y = y

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx : idx + self.seq_len]   # (T, A, F)
        y_next = self.y[idx + self.seq_len]        # (A)

        return X_seq, y_next

# ==========================================
# 2. THE MODEL (The Policy Network)
# ==========================================
class PortfolioNet(nn.Module):
    def __init__(self, num_assets, num_features, hidden_size=64):
        super().__init__()

        self.num_assets = num_assets

        self.lstm = nn.LSTM(
            input_size=num_assets * num_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_assets)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (Batch, Time, Assets, Features)
        B, T, A, F = x.shape

        # Flatten assets + features
        x = x.view(B, T, A * F)

        # LSTM over time
        out, _ = self.lstm(x)

        # Last timestep
        out = out[:, -1, :]   # (Batch, Hidden)

        scores = self.fc(out) # (Batch, Assets)
        weights = self.softmax(scores)

        return weights
# ==========================================
# 3. CUSTOM LOSS FUNCTION (The "SPO" Core)
# ==========================================
def spo_sharpe_loss(weights, future_returns, prev_weights=None, cost_bps=0.0005):
    """
    Differentiable Negative Sharpe Ratio with Transaction Costs.
    """
    # 1. Portfolio Return: dot(w, r)
    # Sum across assets (dim=1)
    port_ret = torch.sum(weights * future_returns, dim=1)
    
    # 2. Transaction Costs (Turnover Penalty)
    # If this is the first batch, assume no previous weights (0 cost)
    if prev_weights is None:
        transaction_cost = 0.0
    else:
        # Turnover = Sum(|w_t - w_{t-1}|)
        turnover = torch.sum(torch.abs(weights - prev_weights), dim=1)
        transaction_cost = turnover * cost_bps
        
    net_ret = port_ret - transaction_cost
    
    # 3. Sharpe Ratio (Mean / Std) across the BATCH
    # We add 1e-6 to sigma to prevent DivisionByZero errors
    expected_ret = torch.mean(net_ret)
    volatility = torch.std(net_ret)
    
    sharpe = expected_ret / (volatility + 1e-6)
    
    # 4. Negate because Optimizer minimizes Loss
    return -sharpe

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
# if __name__ == "__main__":
#     # --- A. Setup Data ---

#     # Create Datasets
#     train_ds = FinancialDataset(X_train, y_train, num_assets=10)
#     test_ds = FinancialDataset(X_test, y_test, num_assets=10)
    
#     # Loaders (Shuffle=False is safer for financial time series debugging)
#     train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
#     test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
#     # --- B. Initialize Model ---
#     model = PortfolioNet(num_features=X_train.shape[1]//10)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     # --- C. Training Loop ---
#     print("\nStarting Training...")
#     epochs = 20
#     train_losses = []
    
#     for epoch in range(epochs):
#         epoch_loss = 0
#         prev_batch_weights = None # Reset for new epoch
        
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
            
#             # Forward Pass: Get Weights
#             curr_weights = model(X_batch)
            
#             # Compute Loss
#             # Note: For simplicity in this demo, we handle prev_weights lightly.
#             # In production, you'd carefully track the last weight of the previous batch.
#             loss = spo_sharpe_loss(curr_weights, y_batch, prev_batch_weights)
            
#             # Backward Pass
#             loss.backward()
#             optimizer.step()
            
#             # Detach weights to use as 'prev' for next step without keeping graph
#             prev_batch_weights = curr_weights.detach()
#             epoch_loss += loss.item()
            
#         avg_loss = epoch_loss / len(train_loader)
#         train_losses.append(avg_loss)
#         if (epoch+1) % 5 == 0:
#             print(f"Epoch {epoch+1}/{epochs} | Negative Sharpe: {avg_loss:.4f}")

#     # --- D. Evaluation (Equity Curve) ---
#     print("\nEvaluating on Test Data...")
#     model.eval()
#     portfolio_values = [1.0] # Start with $1
#     spy_values = [1.0]
    
#     with torch.no_grad():
#         # Iterate through test set day-by-day or batch-by-batch
#         # We process full test set at once for easier plotting
#         X_test_tensor = test_ds.X
#         y_test_tensor = test_ds.y
        
#         # Get Predictions
#         test_weights = model(X_test_tensor)
        
#         # Calculate Returns
#         # (N, Assets) * (N, Assets) -> Sum(dim=1) -> (N,)
#         port_daily_ret = torch.sum(test_weights * y_test_tensor, dim=1).numpy()
        
#         # Calculate Cumulative Return
#         equity_curve = np.cumprod(1 + port_daily_ret)
        
#         # Benchmark (Equal Weight)
#         bench_daily_ret = torch.mean(y_test_tensor, dim=1).numpy()
#         bench_curve = np.cumprod(1 + bench_daily_ret)
        
#     # Plotting
#     plt.figure(figsize=(10,5))
#     plt.plot(equity_curve, label='SPO Model', color='green', linewidth=2)
#     plt.plot(bench_curve, label='Equal Weight Benchmark', color='gray', linestyle='--')
#     plt.title(f"SPO Strategy vs Benchmark (Sharpe: {-train_losses[-1]:.2f})")
#     plt.ylabel("Normalized Portfolio Value")
#     plt.xlabel("Trading Days")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
    
#     print("Done. Copy this logic to your resume!")