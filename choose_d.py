import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load your 1000D dataset
data_1000D =pd.read_csv('data\data\data1000D.csv')

# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(data_1000D, full_matrices=False)

# Calculate the explained variance for each singular value
explained_variance = (S ** 2) / np.sum(S ** 2)

# Create a scree plot to visualize the explained variance
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# Manually choose d based on the scree plot
# Look for the point where the explained variance starts to level off or drop rapidly
# or specify a threshold for explained variance
threshold = 0.95  # Adjust as needed
d = np.argmax(np.cumsum(explained_variance) >= threshold) + 1

# Now, 'd' contains the chosen number of principal components

print(f"Chosen d: {d}")