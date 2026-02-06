<div align="center">

<img width="300" height="300" alt="logo" src="https://github.com/spacedragonx/Predict-then-Optimize-for-Portfolios/blob/main/Graphs/Logo.png?raw=true">

_Optimizing Portfolio Decisions with End-to-End Deep Learning_

<br>

[![Last Commit](https://img.shields.io/badge/last%20commit-recent-blue?style=for-the-badge&labelColor=555555)](https://github.com/spacedragonx/Predict-then-Optimize-for-Portfolios/commits)
[![Python](https://img.shields.io/badge/python-100%25-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=555555)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-pytorch-red?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=555555)](https://pytorch.org/)

<br>

<p>Built with the tools and technologies:</p>

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)](https://matplotlib.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM-blue?style=for-the-badge&labelColor=555555)](#)

</div>


# SPO-LSTM: Smart Portfolio Optimization

A **PyTorch-based deep learning system for end-to-end portfolio allocation**.  
The model uses **LSTM-based market signals** to directly generate portfolio weights, optimizing a **Sharpe-ratio aligned SPO objective** instead of predicting prices.

##  Project Overview
This project formulates portfolio optimization as a **differentiable learning problem**, mapping historical financial features directly to portfolio weights.

**Portfolio weights** represent how capital is split across assets.
A weight of 0.25 means 25% of available capital is allocated to that asset.

The model outputs a valid allocation at every timestep:
- All weights ≥ 0
- All weights sum to 1
- Fully invested, long-only portfolio


##  Why Smart Predict-then-Optimize (SPO)?

Most ML systems in finance optimize the wrong objective:
they minimize prediction error and defer decision-making to a separate optimization step.

This project removes that mismatch.

**Predict-then-Optimize (SPO)** trains the model to directly optimize the *final decision* portfolio weights under the same objective used in production (Sharpe Ratio).


## Key Features

- **End-to-End Optimization:** Learns portfolio allocations directly by optimizing a Sharpe-ratio–aligned objective.
- **Decision-Aware Learning (SPO):** Trains on the final portfolio decision rather than intermediate return predictions.
- **Temporal Modeling:** Uses LSTMs to capture market dynamics and cross-asset relationships over time.
- **Multi-Asset Allocation:** Outputs fully invested, long-only portfolios across multiple assets.
- **Benchmark Evaluation:** Compares performance against an Equal-Weight (EW) baseline.
- **Interpretable Outputs:** Generates dynamic portfolio weights for analyzing allocation shifts and turnover.


##  Decision Policy Network (LSTM → Allocation)

The model acts as a policy network:
given recent market history, it outputs the action to take portfolio weights.

An LSTM encodes temporal market dynamics and cross-asset interactions.
A Softmax layer enforces portfolio constraints by construction.




### Data Pipeline (`FinancialDataset`)
- Raw features are reshaped from `(time, total_features)` into `(time × assets × features)`
- A rolling **20-day window** is used to construct temporal sequences
- Each training sample predicts **next-day returns for all assets**
- Enables the model to condition allocation decisions on recent market history

---

### Policy Network (`PortfolioNet`)
- Market states are flattened across assets and features and passed into an **LSTM**
- The LSTM captures **temporal patterns and cross-asset interactions**
- The final hidden state is projected into asset-level scores
- A **Softmax layer** converts scores into portfolio weights  
  → long-only, fully invested, weights sum to 1

This turns the model into a **direct allocation policy**, not a price predictor.

---

### Decision-Aware Loss (`spo_sharpe_loss`)
Instead of minimizing prediction error, training directly optimizes **risk-adjusted portfolio performance**.

- **Portfolio return:** dot product of predicted weights and realized next-day returns
- **Transaction costs:** modeled via an L1 turnover penalty between consecutive allocations
- **Objective:** minimize the **Negative Sharpe Ratio** over each batch

---

##  Portfolio Weight Dynamics

The model dynamically rebalances the portfolio based on detected market regimes. Below are the weight distributions (Stackplot) for the asset universe over time.

<img width="800" align="center" height='400' alt="logo" src="https://github.com/spacedragonx/Predict-then-Optimize-for-Portfolios/blob/main/Graphs/weights_days_160_220.png?raw=true">

<img width="800" align="center" height='400' alt="logo" src="https://github.com/spacedragonx/Predict-then-Optimize-for-Portfolios/blob/main/Graphs/amzn_close_100_160.png?raw=true">

## Allocation Behavior

The model learns to shift capital based on market regimes rather than maintaining static weights.

In the example below:
- Allocation to AMZN increases ahead of upward price movement
- Exposure is reduced before drawdowns
- Capital is dynamically reallocated to other assets as momentum weakens

This behavior emerges naturally from the objective—no hand-crafted rules.


---
##  Repository Structure
* `models.py`: Contains `PortfolioNet`, `FinancialDataset`, and the `spo_sharpe_loss` function.
* `main.py`: The main training loop, backtesting logic, and plotting.
* `data_save/`: Directory for preprocessed `.pt` tensors.
* `Trained_Models/`: Storage for model checkpoints and state dicts.
* `Graphs/`: Output directory for equity curves and performance metrics.
##  Quick Start

### 1️. Data Download  
Download historical price data for the selected assets (stocks/ETFs) using your preferred data source  
(e.g., Yahoo Finance, Alpaca, or CSV files).

Ensure the raw data includes **OHLCV prices** and is stored in the expected data directory.

---

### 2️. Data Preprocessing  
Run the preprocessing script or notebook to:
- Compute returns and engineered features
- Align assets and timestamps
- Perform temporal train/test split
- Convert data into PyTorch tensors

This step generates the processed dataset:

---

### 3️. Model Training  
Train the SPO-based portfolio optimization model:

```bash
python main.py
```

## Extensions & Production Readiness

This framework is designed to be extensible:
- Plug in alternative objectives (CVaR, downside risk, drawdown-aware loss)
- Add leverage, short-selling, or sector constraints
- Integrate live data feeds for online rebalancing
- Wrap as a service for portfolio recommendation APIs

The core idea**learning decisions instead of predictions** generalizes beyond finance.

## Key Takeaway

This project demonstrates that learning the decision directly instead of predicting prices it can produce more stable, interpretable, and financially aligned portfolio strategies.

SPO-LSTM reframes portfolio optimization as a policy learning problem, bridging modern deep learning with classical portfolio theory.