## Neural State Space Models (NSSM) for Advanced Time Series Forecasting
This repository contains a PyTorch implementation of a Neural State Space Model (NSSM) designed for multivariate time series forecasting. The project demonstrates how to combine the structural advantages of state-space modeling with the flexibility of deep learning to capture complex temporal dynamics and latent patterns.  
## Project Overview
Traditional state-space models often rely on linear assumptions. This project implements a Neural State Space Model where both the transition and observation functions are parameterized by neural networks. This allows the model to learn non-linear transitions in latent space and map those hidden states to observed data points effectively.  
### Key Features
Synthetic Multivariate Dataset: Generates correlated sine waves with evolving noise regimes to simulate real-world data complexity.  
Transition Model: A neural network that predicts the next latent state (s_t) based on the previous state (s_{t-1}) and current input (x_t).  
Observation Model: A neural network that maps the latent state and inputs to the final output (y_t).  
Latent Space Interpretation: Tools to visualize the learned hidden states, providing insight into what the model is "tracking" behind the scenes.  
Benchmarking: Includes a comparison against a Holt-Winters Exponential Smoothing baseline using RMSE.  
## Model Architecture
The NSSM consists of two primary components:
1. Transition Model (T):
   y_t = O(s_t, x_t, x_{t-1})
Uses a sequential architecture with Linear and ReLU layers to evolve the hidden state.
2. Observation Model (O):
   y_t = O(s_t, x_t, x_{t-1})
Decodes the hidden state back into the observation space, utilizing both current and previous time-step inputs for context.
## Getting Started
### Prerequisites
Python 3.x
PyTorch
Pandas / NumPy
Matplotlib
Statsmodels (for baseline comparisons)
### Training Progress
The model is trained using Variational Inference Approximation with an MSE loss function. Initial training results show a consistent decrease in loss:  
Epoch 0: 0.7843
Epoch 20: 0.5656
Epoch 40: 0.3434  
## Results & Visualization
The project includes scripts to visualize the latent dimensions learned during training. These latent states often capture the underlying periodicity and trends of the input signals without being explicitly programmed to do so.  
Baseline Performance:
- Holt-Winters RMSE: 0.5148
