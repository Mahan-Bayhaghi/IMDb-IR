import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import Counter
from Logic.core.clustering.clustering_metrics import *
from Logic.core.clustering.dimension_reduction import *


class ClusteringUtils:
    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.
        max_iter: int
            max_iter of kmeans
        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)
        kmeans.fit(emb_vecs)
        return kmeans.cluster_centers_, kmeans.labels_

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        all_words = [word for document in documents for word in document.split()]
        words_count = Counter(all_words)
        return words_count.most_common(top_n)

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the closest centroid until convergence or a maximum number of iterations is reached. This function uses a random initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(emb_vecs)
        wcss = kmeans.inertia_  # according to sklearn documentation
        return kmeans.cluster_centers_, kmeans.labels_, wcss

    def cluster_hierarchical_single(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage="single")
        clustering.fit(emb_vecs)
        return clustering.labels_

    def cluster_hierarchical_complete(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage="complete")
        clustering.fit(emb_vecs)
        return clustering.labels_

    def cluster_hierarchical_average(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage="average")
        clustering.fit(emb_vecs)
        return clustering.labels_

    def cluster_hierarchical_ward(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage="ward")
        clustering.fit(emb_vecs)
        return clustering.labels_

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name, show_plot=False, plot_to_wandb=True):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        show_plot: boolean
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Perform K-means clustering
        # TODO
        centers, labels = self.cluster_kmeans(emb_vecs=data, n_clusters=n_clusters)

        reduced_data = DimensionReduction().convert_to_2d_tsne(data)

        # Plot the clusters
        # TODO
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, marker='o', cmap='viridis')
        plt.title(f'K-means clustering with {n_clusters} clusters')
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.savefig(f'./figs/{n_clusters}means_clustering.png', dpi=600)
        if show_plot:
            plt.show()
        # Log the plot to wandb
        # TODO
        if plot_to_wandb:
            wandb.log({"K-means Clustering": wandb.Image(plt)})

        # Close the plot display window if needed (optional)
        # TODO
        plt.close()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name, movie_titles, show_plot=False, plot_wandb=True):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        movie_titles
        plot_wandb
        show_plot
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        run = wandb.init(project=project_name, name=run_name)
        # Perform hierarchical clustering
        # TODO
        dend = linkage(data, linkage_method)
        # print(f"data for dend is : {data[:4]}")
        # Create linkage matrix for dendrogram
        # TODO
        plt.figure(figsize=(30, 10), dpi=600)
        dendrogram(dend, labels=movie_titles, leaf_rotation=90)
        plt.title(f'Hierarchical clustering dendrogram ({linkage_method} linkage)')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        if show_plot:
            plt.tight_layout()
            plt.savefig(f'./figs/{linkage_method}.png', dpi=600)
            plt.show()
        # log to wandb !
        if plot_wandb:
            wandb.log({"Hierarchical Clustering Dendrogram": wandb.Image(plt)})
        plt.close()

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None, show_plot=False, plot_to_wandb=True):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        show_plot : boolean
            Show plot to user if true
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        silhouette_scores = []
        purity_scores = []
        # Calculating Silhouette Scores and Purity Scores for different values of k
        for k in k_values:
            # TODO
            centers, labels = self.cluster_kmeans(embeddings, k)
            # Using implemented metrics in clustering_metrics, get the score for each k in k-means clustering
            # and visualize it.
            # TODO
            metrics = ClusteringMetrics()
            # print(f" k means with k = {k} done")
            # print(f"there is {len(centers)} centers and {len(labels)} labels")
            silhouette = metrics.silhouette_score(embeddings, cluster_labels=labels)
            purity = metrics.purity_score(true_labels, cluster_labels=labels)

            silhouette_scores.append(silhouette)
            purity_scores.append(purity)

        # Plotting the scores
        # TODO
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, label='Silhouette Score', marker='o')
        plt.plot(k_values, purity_scores, label='Purity Score', marker='o')
        plt.xlabel('number of clusters (k)')
        plt.ylabel('score')
        plt.title('K-means clustering scores')
        plt.legend()
        plt.savefig(f'./figs/kmeans_scores.png', dpi=600)
        if show_plot:
            plt.show()

        # Logging the plot to wandb
        if plot_to_wandb:
            if project_name and run_name:
                run = wandb.init(project=project_name, name=run_name)
                wandb.log({"Cluster Scores": plt})

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str
                                    , show_plot=False, plot_to_wandb=True):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        show_plot: boolean
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Compute WCSS values for different K values
        wcss_values = []
        for k in k_values:
            # TODO
            centers, labels, wcss = self.cluster_kmeans_WCSS(embeddings, k)
            wcss_values.append(wcss)

        # Plot the elbow method
        # TODO
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, wcss_values, marker='o')
        plt.xlabel('number of clusters (k)')
        plt.ylabel('WCSS')
        plt.title('elbow method for optimal k')
        plt.savefig(f'./figs/elbow.png', dpi=600)
        if show_plot:
            plt.show()
        # Log the plot to wandb
        if plot_to_wandb:
            wandb.log({"Elbow Method": wandb.Image(plt)})
            plt.close()
