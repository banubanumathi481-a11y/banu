# Advanced Time Series Forecasting with Attention-Based LSTMs

## Overview

This project demonstrates **multivariate time series forecasting** using **PyTorch**, comparing a **baseline LSTM** model against an **attention-based LSTM** with built-in explainability. The focus is not only on predictive performance, but also on **interpreting model behavior** via attention weights over historical time steps.

The notebook builds a fully reproducible pipeline: synthetic data generation, preprocessing, model training, evaluation, and attention visualization.

---

## Key Features

* 📈 Multivariate, non-stationary time series generation
* 🧠 Baseline LSTM vs. Attention-based LSTM comparison
* 🔍 Explicit attention mechanism for temporal explainability
* 📊 Forecast evaluation using RMSE and MAE
* 🎯 Visualization of attention weights across lagged inputs

---

## Project Structure

```
Advanced_Time_Series_Forecasting_with_Attention_Based_LSTMs_and_Model_Explainability.ipynb
README.md
```

All implementation and experiments are contained within a single, self‑contained Jupyter Notebook.

---

## Dataset

* **Type:** Synthetic multivariate time series
* **Number of features:** 3
* **Characteristics:**

  * Non-stationary trend
  * Multiple seasonal components
  * Gaussian noise
  * Cross-series dependency

The data is standardized using `StandardScaler` before model training.

---

## Modeling Approach

### 1. Baseline LSTM

* Standard LSTM encoder
* Uses the final hidden state for forecasting
* Serves as a performance reference

### 2. Attention-Based LSTM

* LSTM encoder followed by a **learned attention mechanism**
* Computes importance weights over all time steps
* Produces a context vector as a weighted sum of hidden states
* Improves interpretability by highlighting influential lags

---

## Forecasting Setup

* **Input sequence length:** 30 time steps
* **Forecast horizon:** 5 future steps
* **Train/Test split:** 80% / 20%
* **Loss function:** Mean Squared Error (MSE)
* **Optimizer:** Adam

---

## Evaluation Metrics

The models are evaluated on the test set using:

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**

These metrics allow direct comparison of predictive accuracy between the baseline and attention models.

---

## Explainability with Attention

The attention-based model exposes **attention weights** that quantify the relative importance of each lagged time step.

The notebook includes:

* Visualization of attention weights for individual samples
* Batch-wise extraction of attention patterns
* Analysis of temporal importance across multiple forecasts

This provides insight into *when* the model is focusing in the past to make future predictions.

---

## Requirements

* Python 3.8+
* PyTorch
* NumPy
* Pandas
* scikit-learn
* Matplotlib

Install dependencies with:

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

---

## How to Run

1. Clone the repository
2. Open the notebook:

   ```bash
   jupyter notebook Advanced_Time_Series_Forecasting_with_Attention_Based_LSTMs_and_Model_Explainability.ipynb
   ```
3. Run all cells sequentially

The notebook automatically trains models, evaluates performance, and generates attention visualizations.

---

## Future Extensions

* Apply to real-world datasets (energy, finance, weather)
* Add feature-level (spatial) attention
* Compare with Transformer-based architectures
* Incorporate probabilistic forecasting

---
