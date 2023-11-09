import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def buggy_pca(X, d):
     # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(X)
    
    # Select the top d eigenvectors
    U_d = U[:, :d]
    
    # Project data onto the d-dimensional space
    X_proj = np.dot(X.T, U_d)
    
    # Reconstruct the data in the original space
    X_recon = np.dot(X_proj, U_d.T).T
    
    return X_proj, U_d, X_recon

def demeaned_pca(X, d):
    # Center the data by subtracting the mean of each column
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Perform PCA on the centered data
    X_proj, U_d, X_recon = buggy_pca(X_centered, d)
    X_recon = pd.DataFrame(X_recon , columns = ['7.268685152524053','5.376181293498675']) + X_mean
    return X_proj, U_d, X_recon

def normalized_pca(X, d):
    # Normalize the data (zero mean and unit variance)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std
    
    # Perform PCA on the normalized data
    X_proj, U_d, X_recon = buggy_pca(X_normalized, d)
    X_recon = pd.DataFrame(X_recon , columns = ['7.268685152524053','5.376181293498675']) * X_std + X_mean
    return X_proj, U_d, X_recon



def cost_function(params, X, D, d):
    n = len(X)
    A = params[:D * d].reshape(D, d)
    b = params[D * d:D * d + D]
    Z = params[D * d + D:].reshape(n, d)
    error = X - Z.dot(A.T) - np.outer(np.ones(n), b)
    return np.linalg.norm(error) ** 2

def dro(X, d):
    n = len(X)
    D = len(X.columns)   
    initial_params = np.random.rand(D * d + D + n * d)
    result = minimize(cost_function, initial_params, args=(X,D,d), method='L-BFGS-B')
    optimized_params = result.x

    A = optimized_params[:D * d].reshape(D, d)
    b = optimized_params[D * d:D * d + D]
    Z = optimized_params[D * d + D:].reshape(n, d)

    reconstructed = Z.dot(A.T) + np.outer(np.ones(n), b)
    return Z, optimized_params, reconstructed

data_2d = pd.read_csv('data\data\data2D.csv')
data_1000D = pd.read_csv('data\data\data1000D.csv')
# Define d for the 2D dataset
d_2d = 1
d_1000d = 3

# Perform PCA for all four methods
_,_, reconstructed_demeaned = demeaned_pca(data_2d, d_2d)
_,_,reconstructed_buggy = buggy_pca(data_2d, d_2d)
_,_,reconstructed_normalized = normalized_pca(data_2d, d_2d)
_,_,reconstructed_dro = dro(data_2d, d_2d)


_,_,reconstructed_buggy1000 = buggy_pca(data_1000D, d_1000d)


# Calculate reconstruction errors for all methods
def calculate_reconstruction_error(original, reconstructed):
    return np.sum((original - reconstructed) ** 2)

error_demeaned = calculate_reconstruction_error(data_2d, reconstructed_demeaned)
error_buggy = calculate_reconstruction_error(data_2d, reconstructed_buggy)
error_normalized = calculate_reconstruction_error(data_2d, reconstructed_normalized)
error_dro = calculate_reconstruction_error(data_2d, reconstructed_dro)

# Create a plot comparing the original and reconstructed points for all four methods
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.scatter(data_2d.iloc[:, 0], data_2d.iloc[:, 1], label="Original")
plt.scatter(reconstructed_demeaned.iloc[:, 0], reconstructed_demeaned.iloc[:, 1], label="Reconstructed")
plt.title("Demeaned PCA")
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(data_2d.iloc[:, 0], data_2d.iloc[:, 1], label="Original")
plt.scatter(reconstructed_buggy[:, 0], reconstructed_buggy[:, 1], label="Reconstructed")
plt.title("Buggy PCA")
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(data_2d.iloc[:, 0], data_2d.iloc[:, 1], label="Original")
plt.scatter(reconstructed_normalized.iloc[:, 0], reconstructed_normalized.iloc[:, 1], label="Reconstructed")
plt.title("Normalized PCA")
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(data_2d.iloc[:, 0], data_2d.iloc[:, 1], label="Original")
plt.scatter(reconstructed_dro[:, 0], reconstructed_dro[:, 1], label="Reconstructed")
plt.title("DRO PCA")
plt.legend()


plt.show()

# Report reconstruction errors
print("Reconstruction Error (Demeaned PCA):", error_demeaned)
print("Reconstruction Error (Buggy PCA):", error_buggy)
print("Reconstruction Error (Normalized PCA):", error_normalized)
print("Reconstruction Error (DRO PCA):", error_dro)











