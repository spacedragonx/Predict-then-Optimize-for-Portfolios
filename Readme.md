# SPO-LSTM: Smart Portfolio Optimization

A PyTorch-based Deep Learning framework that implements **Smart Portfolio Optimization (SPO)**. Unlike traditional methods that predict prices first, this model uses an **LSTM (Long Short-Term Memory)** network to directly output optimal asset weights by maximizing risk-adjusted returns (Sharpe Ratio).

[Image of LSTM architecture for time series forecasting]

## 🎯 Project Overview
This project treats portfolio optimization as a differentiable end-to-end learning problem. The model learns to map historical financial features directly to portfolio weights $w$, optimizing a custom loss function based on the negative Sharpe Ratio.

## 🚀 Key Features
* **End-to-End Optimization:** Direct optimization of the Sharpe Ratio via `spo_sharpe_loss`.
* **Temporal Modeling:** Uses LSTM layers to capture multi-asset correlations and time-series dependencies.
* **Leakage-Free Validation:** Implements a chronological 80/20 split to ensure realistic backtesting.
* **Automated Benchmarking:** Real-time comparison against an **Equal-Weight (EW)** strategy.
* **PyTorch 2.6+ Ready:** Uses `weights_only=True` for secure data loading.

## 🛠️ Tech Stack
* **Core:** Python, PyTorch
* **Data Handling:** NumPy, Torch DataLoader
* **Visualization:** Matplotlib
* **Optimization:** Adam Optimizer

## 📁 Repository Structure
* `models.py`: Contains `PortfolioNet`, `FinancialDataset`, and the `spo_sharpe_loss` function.
* `train_spo.py`: The main training loop, backtesting logic, and plotting.
* `data_save/`: Directory for preprocessed `.pt` tensors.
* `Trained_Models/`: Storage for model checkpoints and state dicts.
* `Graphs/`: Output directory for equity curves and performance metrics.

## ⚙️ How It Works

### 1. The Model Architecture
The `PortfolioNet` processes a sequence of features (default `SEQ_LEN = 20`) for all tickers. It outputs a weight vector that is constrained to sum to 1.

[Image of Sharpe ratio formula and efficient frontier]

### 2. Loss Function
The model minimizes the **Negative Sharpe Ratio**:
$$Loss = -\frac{E[R_p]}{\sigma(R_p)}$$
where $R_p$ is the portfolio return calculated as $\sum (weights \times returns)$.

## 📊 Quick Start

1. **Prepare Data:** Ensure your processed tensor is at `data_save/25_spo_data.pt`.
2. **Run Training:**
   ```bash
   python train_spo.py