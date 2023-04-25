import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Step 1: Load the feature vectors
with open('embeddingscnnmodelfinal(6layer).pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Step 2: Create a t-SNE model with desired hyperparameters
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)

# Step 3: Fit the t-SNE model on the feature vectors
reduced_embeddings = tsne.fit_transform(embeddings)

# Step 4: Visualize the reduced-dimensional embeddings using a scatter plot
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
plt.show()
