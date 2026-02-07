# Advanced Time Series Forecasting with Attention-Based LSTMs
This project implements and compares a Baseline LSTM model with an Attention-Based LSTM for multivariate time series forecasting. It includes a synthetic data generation pipeline, custom PyTorch datasets, and an explainability component to visualize how the model prioritizes different time steps.  
# Project Overview
The goal of this project is to forecast future values in a non-stationary dataset using deep learning. By introducing an explicit Attention Mechanism, the model can assign varying levels of "importance" to different lagged time steps, providing a layer of interpretability often missing in standard recurrent neural networks.  
# Features
Synthetic Data Generation: Creates a non-stationary dataset with trend, multiple seasonality components, and Gaussian noise.  
Dual Architecture Comparison:
Baseline LSTM: A standard many-to-one architecture.  
Attention LSTM: Uses a linear layer to calculate alignment scores and context vectors across all hidden states.  
Explainability: Visualizes attention weights to show which lagged time steps (e.g., more recent vs. older data) most influence the forecast.  
Evaluation Metrics: Computes Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) for performance assessment.  
# Model Performance
Based on the experimental results provided in the notebook, the two models performed as follows:
# Model          RMSE    MAE
Baseline LSTM ~0.1886 ~0.1511
Attention LSTM ~0.2005 ~0.1594
# Technical Implementation
# Data Specifications
Sequence Length (SEQ\_LEN): 30 (The model looks at the previous 30 time steps).  
Forecast Horizon (HORIZON): 5 (The model predicts the next 5 time steps).  
Input Dimensions: 3 (Multivariate features).  
# Attention Mechanism Logic
The attention weights are calculated by passing all LSTM hidden states through a linear layer, followed by a Softmax function to ensure the importance scores sum to 1.  
The context vector is then produced as a weighted sum:
   Context = \sum_{i=1}^{n} weights_i \times lstm\_outputs_i
# Requirements
Python 3.x
PyTorch
NumPy & Pandas
Scikit-learn
Matplotlib
