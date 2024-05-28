import json

import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils
import Logic.core.path_access as path_access

# Main Function: Clustering Tasks

fasttext_model = FastText()
# preprocessed_path = path_access.path_to_logic() + "../IMDB_crawled_preprocessed.json"
preprocessed_path = path_access.path_to_logic() + "core/word_embedding/training_data.csv"
data_loader = FastTextDataLoader(file_path=preprocessed_path)
X, y = data_loader.create_train_data_for_cluster()
print("here")
fasttext_model.train(X)
fasttext_model.prepare(dataset_path=None, mode='save', save=True, path='./Fasttext_model_clustering.bin')

# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
embeddings = []
for i in range(len(X)):
    embeddings.append(fasttext_model.get_query_embedding(X[i]).tolist())
with open('./clustering_embeddings.json', 'w') as outfile:
    json.dump(embeddings, outfile)
with open('./clustering_labels.json', 'w') as outfile:
    json.dump(y.tolist(), outfile)

# 1. Dimension Reduction
# TODO: Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.
dimension_reducer = DimensionReduction()
reduced_embeddings = dimension_reducer.pca_reduce_dimension(embeddings, n_components=2)
svs = dimension_reducer.pca.singular_values_
explained_variance_ratio = dimension_reducer.pca.explained_variance_ratio_
print(f"Singular values are : {svs}")
print(f"Explained variance ratio is : {explained_variance_ratio}")

# TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.

clustering_metrics = ClusteringMetrics()
clustering_utils = ClusteringUtils()

# 2. Clustering
## K-Means Clustering
# TODO: Implement the K-means clustering algorithm from scratch.
# TODO: Create document clusters using K-Means.
# TODO: Run the algorithm with several different values of k.
# TODO: For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)

## Hierarchical Clustering
# TODO: Perform hierarchical clustering with all different linkage methods.
# TODO: Visualize the results.

# 3. Evaluation
# TODO: Using clustering metrics, evaluate how well your clustering method is performing.
