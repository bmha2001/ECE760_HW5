import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic data
def generate_synthetic_data(sigma):
    np.random.seed(0)  # For reproducibility
    num_points = 100
    mean_a = [-1, -1]
    mean_b = [1, -1]
    mean_c = [0, 1]
    cov_a = [[sigma, 0.5], [0.5, 1]]
    cov_b = [[1, -0.5], [-0.5, 2]]
    cov_c = [[sigma, 0], [0, 2]]

    data_a = np.random.multivariate_normal(mean_a, cov_a, num_points)
    data_b = np.random.multivariate_normal(mean_b, cov_b, num_points)
    data_c = np.random.multivariate_normal(mean_c, cov_c, num_points)

    synthetic_data = np.vstack((data_a, data_b, data_c))

    return synthetic_data

# K-means clustering
def kmeans(data, k, max_iters=100):
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels

# GMM using Expectation-Maximization
def em_gmm(data, k, max_iters=100):
    n, dim = data.shape
    # Initialize parameters
    weights = np.ones(k) / k
    means = data[np.random.choice(n, k, replace=False)]
    covariances = [np.eye(dim) for _ in range(k)]

    for _ in range(max_iters):
        # Expectation step
        responsibilities = np.zeros((n, k))
        for j in range(k):
            responsibilities[:, j] = weights[j] * multivariate_normal(data, means[j], covariances[j])

        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]

        # Maximization step
        Nk = responsibilities.sum(axis=0)
        weights = Nk / n
        means = (data[:, np.newaxis] * responsibilities[:, :, np.newaxis]).sum(axis=0) / Nk[:, np.newaxis]
        covariances = [(responsibilities[:, j, np.newaxis] * (data - means[j])).T @ (data - means[j]) / Nk[j] for j in range(k)]

    return weights, means, covariances, responsibilities

# Multivariate Gaussian PDF
def multivariate_normal(x, mean, cov):
    dim = len(mean)
    det = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_const = 1.0 / ((2 * np.pi) ** (dim / 2) * np.sqrt(det))
    x_mean = x - mean
    exponent = -0.5 * np.sum(x_mean @ inv_cov * x_mean, axis=1)
    return norm_const * np.exp(exponent)

# Clustering objective: Sum of squared distances
def clustering_objective(data, centroids, labels):
    distances = np.linalg.norm(data - centroids[labels], axis=1)
    return np.sum(distances**2)

# Clustering accuracy: Accuracy of clustering
def clustering_accuracy(true_labels, predicted_labels):
    n = len(true_labels)
    correct = sum(true_labels == predicted_labels)
    return correct / n

# Main code
sigma_values = [0.5, 1, 2, 4, 8]
k_values = 3
num_datasets = 5
num_runs = 10  # Number of runs for each dataset and method

kmeans_objectives = np.zeros((num_datasets, num_runs, len(sigma_values)))
kmeans_accuracies = np.zeros((num_datasets, num_runs, len(sigma_values)))
em_gmm_objectives = np.zeros((num_datasets, num_runs, len(sigma_values)))
em_gmm_accuracies = np.zeros((num_datasets, num_runs, len(sigma_values)))

for i, sigma in enumerate(sigma_values):
    for j in range(num_datasets):
        synthetic_data = generate_synthetic_data(sigma)
        
        for run in range(num_runs):
            # K-means
            centroids, kmeans_labels = kmeans(synthetic_data, k_values)
            kmeans_objectives[j, run, i] = clustering_objective(synthetic_data, centroids, kmeans_labels)
            true_labels = np.concatenate([np.full(100, 0), np.full(100, 1), np.full(100, 2)])
            kmeans_accuracies[j, run, i] = clustering_accuracy(true_labels, kmeans_labels)
            
            # GMM
            weights, means, covariances, _ = em_gmm(synthetic_data, k_values)
            em_gmm_labels = np.argmax(weights, axis=0)
            em_gmm_objectives[j, run, i] = clustering_objective(synthetic_data, means, em_gmm_labels)
            em_gmm_accuracies[j, run, i] = clustering_accuracy(true_labels, em_gmm_labels)

# Plot results
for metric_name, metric_values, ylabel in [
    ("Kmeans Clustering Objective", kmeans_objectives, "Sum of Squared Distances"),
    ("Kmeans Clustering Accuracy", kmeans_accuracies, "Accuracy"),
    ("GMM Clustering Objective", em_gmm_objectives, "Log-Likelihood"),
    ("GMM Clustering Accuracy", em_gmm_accuracies, "Accuracy")
]:
    for i in range(len(sigma_values)):
        plt.figure(figsize=(8, 6))
        plt.plot(sigma_values, metric_values.mean(axis=(0, 1)), marker='o')
        plt.xlabel('σ (Sigma)')
        plt.ylabel(ylabel)
        plt.title(f'{metric_name} vs. Sigma for K-means and GMM')
        plt.xticks(sigma_values)
        plt.grid()
        plt.show()
