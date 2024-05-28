import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

import Logic.core.path_access as path_access
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText

if __name__ == '__main__':
    # Main Function: Clustering Tasks
    fasttext_model = FastText()
    preprocessed_path = path_access.path_to_logic() + "IMDB_crawled_preprocessed.json"
    data_loader = FastTextDataLoader(file_path=preprocessed_path)
    X, y, movie_titles = data_loader.create_train_data_for_cluster(size=None)
    with open('./clustering_text.txt', 'w', encoding='utf-8') as f:
        for line in X:
            f.write(str(line))
            f.write("\n")
    print("clustering train data loaded and saved to file")
    fasttext_model.train(texts_path='./clustering_text.txt', epochs=5, dimension=100)
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
        np.array(tsne_reduced_embeddings), project_name='sample name', run_name='run name', show_plot=True,
        plot_wandb=False)
    dimension_reducer.wandb_plot_explained_variance_by_components(np.array(reduced_embeddings),
                                                                  project_name='sample name',
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
                                                k_values, project_name='k means', run_name='run', show_plot=True,
                                                plot_to_wandb=False)
    print("clustering scores plotted")
    clustering_utils.visualize_elbow_method_wcss(reduced_embeddings, k_values,
                                                 project_name='elbow visualization', run_name='run', show_plot=True,
                                                 plot_to_wandb=False)
    print("elbow method plotted")
    clustering_utils.visualize_kmeans_clustering_wandb(np.array(reduced_embeddings), 5,
                                                       project_name="k means visualization",
                                                       run_name='run', show_plot=True, plot_to_wandb=False)
    print("k means visualized")

    ## Hierarchical Clustering
    max_visualization_threshold = 200
    movie_titles = movie_titles[:min(len(movie_titles), max_visualization_threshold)]

    linkage_methods = ['single', 'complete', 'average', 'ward']
    for linkage_method in linkage_methods:
        (clustering_utils.wandb_plot_hierarchical_clustering_dendrogram
         (np.array(reduced_embeddings)[:max_visualization_threshold], linkage_method=linkage_method,
          project_name='clustering', run_name=f'{linkage_method} hierarchical clustering'
          , show_plot=True, plot_wandb=False, movie_titles=movie_titles))
        print(f"linkage methode {linkage_method} plotted")

    # 3. Evaluation
    print("----" * 25)
    print(f"K-means methods with k = {7}")
    cluster_centers, cluster_labels = clustering_utils.cluster_kmeans(embeddings, 5)
    silhouette_score = clustering_metrics.silhouette_score(embeddings, cluster_labels)
    print(f"silhouette_score is {silhouette_score}")
    purity_score = clustering_metrics.purity_score(encoded_labels, cluster_labels)
    print(f"purity_score is {purity_score}")
    rand_score = clustering_metrics.adjusted_rand_score(encoded_labels, cluster_labels)
    print(f"rand_score is {rand_score}")

    # hierarchical clusterings
    methods = [clustering_utils.cluster_hierarchical_single, clustering_utils.cluster_hierarchical_complete, clustering_utils.cluster_hierarchical_average, clustering_utils.cluster_hierarchical_ward]
    for method in methods:
        print("----" * 25)
        print(f"linkage methode : {linkage_methods[methods.index(method)]}")
        cluster_labels = method(reduced_embeddings)
        silhouette_score = clustering_metrics.silhouette_score(reduced_embeddings, cluster_labels)
        print(f"silhouette_score is {silhouette_score}")
        purity_score = clustering_metrics.purity_score(encoded_labels, cluster_labels)
        print(f"purity_score is {purity_score}")
        rand_score = clustering_metrics.adjusted_rand_score(encoded_labels, cluster_labels)
        print(f"rand_score is {rand_score}")

# run test #
# len of movies : 9950
# clustering train data loaded and saved to file
# Read 44M words
# Number of words:  71604
# Number of labels: 0
# Progress: 100.0% words/sec/thread:   27725 lr:  0.000000 avg.loss:  1.514569 ETA:   0h 0m 0s
# embeddings and labels saved
# embeddings reduced (PCA)
# embeddings reduced (t-SNE)
# wandb: Currently logged in as: marlicartworks. Use `wandb login --relogin` to force relogin
# wandb: Tracking run with wandb version 0.17.0
# wandb: Run data is saved locally in D:\Sharif\Daneshgah stuff\term 6\mir\project\IMDb-IR\Logic\core\clustering\wandb\run-20240528_210415-6fvvyy92
# wandb: Run `wandb offline` to turn off syncing.
# wandb: Syncing run run name
# wandb:  View project at https://wandb.ai/marlicartworks/sample%20name
# wandb:  View run at https://wandb.ai/marlicartworks/sample%20name/runs/6fvvyy92
# clustering labels loaded
# ground_truth labels saved
# encoded labels saved
# k means visualized
# linkage methode single plotted
# linkage methode complete plotted
# linkage methode average plotted
# linkage methode ward plotted
# ----------------------------------------------------------------------------------------------------
# K-means methods with k = 7
# silhouette_score is 0.1235496275180084
# purity_score is 0.23547738693467338
# rand_score is 0.02992408180473677
# ----------------------------------------------------------------------------------------------------
# linkage methode : single
# silhouette_score is 0.9264934648828059
# purity_score is 0.21306532663316582
# rand_score is 0.005139838955819241
# ----------------------------------------------------------------------------------------------------
# linkage methode : complete
# silhouette_score is 0.9264934648828059
# purity_score is 0.21306532663316582
# rand_score is 0.005139838955819241
# ----------------------------------------------------------------------------------------------------
# linkage methode : average
# silhouette_score is 0.9264934648828059
# purity_score is 0.21306532663316582
# rand_score is 0.005139838955819241
# ----------------------------------------------------------------------------------------------------
# linkage methode : ward
# silhouette_score is 0.9264934648828059
# purity_score is 0.21306532663316582
# rand_score is 0.005139838955819241

