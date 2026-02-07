# Advanced Time Series Forecasting with Attention-Based LSTMs

This project demonstrates an **end-to-end deep learning pipeline for multivariate time series forecasting** using PyTorch. It compares a **baseline LSTM** against an **Attention-enhanced LSTM** and includes a simple **model explainability** component via attention weight visualization.

The notebook is fully self-contained and uses a **synthetic, non-stationary multivariate dataset**, making it ideal for learning, experimentation, and extension to real-world data.

---

## 📌 Key Features

* Synthetic **multivariate, non-stationary time series** generation
* Data preprocessing with **standardization**
* Custom **PyTorch Dataset & DataLoader** for sequence-to-multi-step forecasting
* **Baseline LSTM** for comparison
* **Attention-based LSTM** with explicit temporal attention mechanism
* Multi-step forecasting (**horizon > 1**)
* Evaluation using **RMSE** and **MAE**
* **Explainability** via attention weight visualization
* Reproducible results with fixed random seeds

---

## 📂 Project Structure

```
Advanced_Time_Series_Forecasting_with_Attention_Based_LSTMs_and_Model_Explainability.ipynb
README.md
```

All logic (data generation, models, training, evaluation, and explainability) is contained within the notebook.

---

## ⚙️ Requirements

Install the following Python packages before running the notebook:

```bash
pip install numpy pandas torch scikit-learn matplotlib
```

> **Optional:** CUDA-enabled GPU for faster training (automatically detected by PyTorch).

---

## 🧠 Methodology Overview

### 1. Data Generation

* Creates a **3-variable time series** with:

  * Trend
  * Multiple seasonal components
  * Gaussian noise
* One variable is lagged and correlated with another to simulate real-world dependencies.

### 2. Preprocessing

* Standardization using `StandardScaler`
* Sliding window approach:

  * **Input sequence length:** 30 time steps
  * **Forecast horizon:** 5 future steps (targeting variable 1)

### 3. Models

#### Baseline LSTM

* Standard LSTM
* Uses the **last hidden state** for prediction
* Serves as a performance benchmark

#### Attention-Based LSTM

* LSTM encoder over the input sequence
* **Temporal attention mechanism** learns importance weights over time steps
* Context vector is used for final prediction

### 4. Training

* Optimizer: **Adam**
* Loss function: **Mean Squared Error (MSE)**
* Training epochs: 15
* Batch size: 32

### 5. Evaluation

* Metrics:

  * **RMSE (Root Mean Squared Error)**
  * **MAE (Mean Absolute Error)**
* Comparison between baseline and attention models

---

## 📊 Explainability: Attention Weights

The attention-based model provides **interpretability** by highlighting which historical time steps contribute most to the forecast.

* Attention weights are extracted from the model
* A line plot visualizes **importance across lagged inputs**
* Helps explain *why* the model makes certain predictions

---

## ▶️ How to Run

1. Open the notebook:

   ```bash
   jupyter notebook Advanced_Time_Series_Forecasting_with_Attention_Based_LSTMs_and_Model_Explainability.ipynb
   ```
2. Run all cells sequentially
3. Observe:

   * Training results
   * Performance comparison
   * Attention weight visualization

---

## 🚀 Possible Extensions

* Replace synthetic data with **real-world datasets** (finance, energy, sensors)
* Add **feature-level attention**
* Compare with GRU, Transformer, or Temporal CNNs
* Use **probabilistic forecasting** (quantile loss)
* Integrate SHAP or Integrated Gradients for deeper explainability

---
