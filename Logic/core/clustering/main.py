import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

import Logic.core.path_access as path_access
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText

# Main Function: Clustering Tasks
fasttext_model = FastText()
preprocessed_path = path_access.path_to_logic() + "IMDB_crawled_preprocessed.json"
# preprocessed_path = path_access.path_to_logic() + "core/word_embedding/my_csv_small_small_small.csv"
data_loader = FastTextDataLoader(file_path=preprocessed_path)
X, y, movie_titles = data_loader.create_train_data_for_cluster(size=1000)
with open('./clustering_text.txt', 'w', encoding='utf-8') as f:
    for line in X:
        f.write(str(line))
        f.write("\n")
print("clustering train data loaded and saved to file")
fasttext_model.train(texts_path='./clustering_text.txt')
fasttext_model.prepare(dataset_path=None, mode='save', save=True, path='./Fasttext_model_clustering.bin')

# 0. Embedding Extraction
embeddings = []
for i in range(len(X)):
    embeddings.append(fasttext_model.get_query_embedding(str(X[i])).tolist())
with open('./clustering_embeddings.json', 'w') as outfile:
    json.dump(embeddings, outfile)
with open('./clustering_labels.json', 'w') as outfile:
    json.dump(y, outfile)
print("embeddings and labels saved")

# 1. Dimension Reduction
dimension_reducer = DimensionReduction()
reduced_embeddings = dimension_reducer.pca_reduce_dimension(np.array(embeddings), n_components=2)
print("embeddings reduced (PCA)")

tsne_reduced_embeddings = dimension_reducer.convert_to_2d_tsne(np.array(embeddings))
print("embeddings reduced (t-SNE)")
dimension_reducer.wandb_plot_2d_tsne(
    np.array(tsne_reduced_embeddings), project_name='sample name', run_name='run name', show_plot=True, plot_wandb=False)
dimension_reducer.wandb_plot_explained_variance_by_components(np.array(reduced_embeddings), project_name='sample name',
                                                              run_name='run name', show_plot=True, plot_wandb=False)

# 2. Clustering
## K-Means Clustering
clustering_metrics = ClusteringMetrics()
clustering_utils = ClusteringUtils()

with open('./clustering_labels.json', 'r') as infile:
    clustering_labels = json.load(infile)
print(f"clustering labels loaded")

ground_truth = [labels.split()[0] if len(labels) != 0 else 'N/A' for labels in clustering_labels]
with open('./ground_truth_labels.json', 'w') as outfile:
    json.dump(ground_truth, outfile)
print(f"ground_truth labels saved")

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(ground_truth)
with open('./encoded_labels.json', 'w') as outfile:
    json.dump(encoded_labels.tolist(), outfile)
print(f"encoded labels saved")

k_values = [i for i in range(2, 10)]
clustering_utils.plot_kmeans_cluster_scores(reduced_embeddings, encoded_labels.tolist(),
                                            k_values, project_name='k means', run_name='run', show_plot=True, plot_to_wandb=False)
print("clustering scores plotted")
clustering_utils.visualize_elbow_method_wcss(reduced_embeddings, k_values,
                                             project_name='elbow visualization', run_name='run', show_plot=True, plot_to_wandb=False)
print("elbow method plotted")
clustering_utils.visualize_kmeans_clustering_wandb(np.array(reduced_embeddings), 5, project_name="k means visualization",
                                                   run_name='run', show_plot=True, plot_to_wandb=False)
print("k means visualized")

## Hierarchical Clustering
max_visualization_threshold = 100
movie_titles = movie_titles[:min(len(movie_titles), max_visualization_threshold)]
print(f"movie titles is : {movie_titles}")

linkage_methods = ['single', 'complete', 'average', 'ward']
for linkage_method in linkage_methods:
    (clustering_utils.wandb_plot_hierarchical_clustering_dendrogram
     (np.array(reduced_embeddings)[:max_visualization_threshold], linkage_method=linkage_method, project_name='clustering', run_name=f'{linkage_method} hierarchical clustering'
      , show_plot=True, plot_wandb=False, movie_titles=movie_titles))
    print(f"linkage methode {linkage_method} plotted")

# 3. Evaluation
# TODO: Using clustering metrics, evaluate how well your clustering method is performing.
