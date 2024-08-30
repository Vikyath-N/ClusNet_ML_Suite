# ClusNet ML Suite

## Overview

This project encompasses the implementation of the Expectation-Maximization (EM) algorithm for Gaussian Mixture Models (GMM) and K-Means clustering. Additionally, it includes the simulation and training of a neural network using backpropagation.

## Files

- `ML_proj.py`: Contains the implementation of the EM algorithm, K-Means clustering, and neural network training.
- `README.md`: This file with usage instructions and license information.

## Usage Instructions

1. **Clustering (EM Algorithm and K-Means)**
   - Run the `main_clustering()` function to execute the clustering algorithms.
   - Visualize the results for different values of k (number of clusters) for both GMM and K-Means.

2. **Neural Network Simulation and Training**
   - Run the `main_neural_network()` function to simulate and train the neural network.
   - The network is trained on a given dataset using stochastic gradient descent.
   - The trained weights are displayed, and the loss over epochs is plotted.

## Dependencies

- numpy
- matplotlib
- scipy
- sklearn

You can install the required packages using pip:

```sh
pip install numpy matplotlib scipy scikit-learn
