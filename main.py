# main.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Function to get initial cluster parameters for GMM
def get_initial_parameters(k, X):
    pi = np.full(shape=k, fill_value=1/k) 
    random_row = np.random.randint(low=0, high=X.shape[0], size=k)
    mu = [X[row_index, :] for row_index in random_row] 
    sigma = [np.cov(X.T) for _ in range(k)] 
    return pi, mu, sigma

# E-step of the EM algorithm
def e_step(data, mu, sigma, pi, k):
    n_points = data.shape[0]
    weights = np.zeros((n_points, k))
    for i in range(k):
        distribution = multivariate_normal(mean=mu[i], cov=sigma[i])
        weights[:, i] = distribution.pdf(data)
    likelihood = np.dot(weights, pi)
    weights = weights * pi / likelihood[:, np.newaxis]
    return weights

# M-step of the EM algorithm
def m_step(data, mu, sigma, pi, weights, k):
    n_points = data.shape[0]
    dim = data.shape[1]
    for i in range(k):
        weight_sum = np.sum(weights[:, i])
        mu[i] = np.sum(data * weights[:, i, np.newaxis], axis=0) / weight_sum
        diff = data - mu[i]
        sigma[i] = np.dot(weights[:, i] * diff.T, diff) / weight_sum
        pi[i] = weight_sum / n_points
    return mu, sigma, pi

# GMM implementation
def gmm(data, mu, sigma, pi, k, max_iterations=1000):
    for _ in range(max_iterations):
        weights = e_step(data, mu, sigma, pi, k)
        mu, sigma, pi = m_step(data, mu, sigma, pi, weights, k)
    weights = e_step(data, mu, sigma, pi, k)
    assignments = np.argmax(weights, axis=1)
    return mu, sigma, pi, assignments

# Function to get initial centroids for K-means
def get_initial_clusters(k, X):
    random_indices = np.random.choice(X.shape[0], k, replace=False)
    initial_centroids = X[random_indices]
    return initial_centroids

# Update assignments for K-means
def update_assignments(data, centroids):
    assignments = np.zeros(data.shape[0], dtype=int)
    for i in range(data.shape[0]):
        distances = np.linalg.norm(data[i] - centroids, axis=1)
        assignments[i] = np.argmin(distances)
    return assignments

# Update centroids for K-means
def update_centroids(data, assignments, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster = data[assignments == i]
        if cluster.shape[0] > 0:
            centroids[i] = np.mean(cluster, axis=0)
    return centroids

# K-means implementation
def kmeans(data, initial_centroids, max_iterations=100):
    centroids = initial_centroids
    for iteration in range(max_iterations):
        assignments = update_assignments(data, centroids)
        new_centroids = update_centroids(data, assignments, initial_centroids.shape[0])
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    return centroids, assignments

# Main code for clustering
def main_clustering():
    n_samples = 4000
    n_components = 4
    X, y_true = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0)
    colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]

    for k in [3, 4]:
        pi_initial, mu_initial, sigma_initial = get_initial_parameters(k, X)
        mu_final, sigma_final, pi_final, gmm_cluster_assignments = gmm(X, mu_initial, sigma_initial, pi_initial, k)
        plt.figure(figsize=(8, 6))
        for k_index, col in enumerate(colors[:k]):
            cluster_data = gmm_cluster_assignments == k_index
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)
        plt.title(f'GMM Clustering with k={k}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

        initial_centroids = get_initial_clusters(k, X)
        centroids, kmeans_cluster_assignments = kmeans(X, initial_centroids, 10)
        plt.figure(figsize=(8, 6))
        for k_index, col in enumerate(colors[:k]):
            cluster_data = X[kmeans_cluster_assignments == k_index]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=col, marker=".", s=10)
            plt.scatter(centroids[k_index, 0], centroids[k_index, 1], c="red", marker="x", s=100)
        plt.title(f'K-Means Clustering with k={k}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

# Simulate neural network
def simulate_neural_network(weights, input_value):
    outputs = [None, None, None, None, input_value]
    for i in range(4):
        a_i = np.sum(weights[i] * outputs)
        outputs[i] = sigmoid(a_i)
    return outputs

# Calculate deltas for backpropagation
def calculate_deltas(z, target, weights):
    delta_1 = (z[0] - target) * sigmoid_derivative(z[0])
    deltas_hidden = [weights[i][0] * delta_1 * sigmoid_derivative(z[i+1]) for i in range(3)]
    delta_5 = sum(weights[i][4] * deltas_hidden[i] for i in range(3))
    return [delta_1] + deltas_hidden + [delta_5]

# Backpropagation function
def backward_pass(X, y, output, hiddenLayer_activations, weights_input_hidden, weights_hidden_output, lr):
    error = y - output
    d_output = error * sigmoid_derivative(output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hiddenLayer_activations)
    X = X.reshape(-1, 1)
    weights_hidden_output_update = hiddenLayer_activations.T.dot(d_output) * lr
    weights_input_hidden_update = X.T.dot(d_hidden_layer) * lr
    return weights_input_hidden_update, weights_hidden_output_update

# Training neural network function
def train_neural_network(X_train, y_train, lr=0.1, epochs=1000):
    inputLayer_neurons = X_train.shape[1]
    hiddenLayer_neurons = 3
    outputLayer_neurons = 1
    weights_input_hidden = np.random.uniform(size=(inputLayer_neurons, hiddenLayer_neurons))
    weights_hidden_output = np.random.uniform(size=(hiddenLayer_neurons, outputLayer_neurons))
    losses = []
    X_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    for epoch in range(epochs):
        output, hiddenLayer_activations = forward_pass(X_train, weights_input_hidden, weights_hidden_output)
        weights_input_hidden_update, weights_hidden_output_update = backward_pass(
            X_train, y_train, output, hiddenLayer_activations, weights_input_hidden, weights_hidden_output, lr
        )
        weights_input_hidden -= weights_input_hidden_update
        weights_hidden_output -= weights_hidden_output_update
        loss = np.mean((y_train - output) ** 2)
        losses.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.5f}")
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.show()
    return weights_input_hidden, weights_hidden_output

# Main code for neural network training
def main_neural_network():
    X_train = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    y_train = np.array([0.7312, 0.7339, 0.7438, 0.7832, 0.8903, 0.9820, 0.8114, 0.5937, 0.5219, 0.5049, 0.5002])
    X_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    weights_input_hidden, weights_hidden_output = train_neural_network(X_train, y_train)
    print("Trained weights from input to hidden layer:\n", weights_input_hidden)
    print("Trained weights from hidden to output layer:\n", weights_hidden_output)

if __name__ == "__main__":
    main_clustering()
    main_neural_network()
