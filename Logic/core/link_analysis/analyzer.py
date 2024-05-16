import json
import random

import networkx as nx

from Logic.core.link_analysis.graph import LinkGraph
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader
import Logic.core.path_access as path_access
from random import sample

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []  # stars are hubs
        self.authorities = []   # ids are authorities
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        # ido = self.root_set[0]['id']
        for movie in self.root_set:
            # TODO
            movie_id = movie["id"]
            self.graph.add_node(movie_id)
            if movie["stars"] != "N/A":
                for star in movie["stars"]:
                    self.graph.add_node(star)
                    self.graph.add_edge(star, movie_id)
        # nx.draw(self.graph)
        # print(f"{self.graph.get_predecessors(ido)}")
        print("root set initiated")

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in self.root_set:
            # TODO
            movie_id = movie["id"]
            self.graph.add_node(movie_id)
            if movie_id not in self.authorities:
                self.authorities.append(movie_id)

            if movie["stars"] != "N/A":
                for star in movie["stars"]:
                    self.graph.add_node(star)
                    self.graph.add_edge(star, movie_id)
                    if star not in self.hubs:
                        self.hubs.append(star)

        print(f"authorities are {self.authorities}")
        print(f"hubs are {self.hubs}")

    def hits_(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = []
        h_s = []

        # TODO
        # for node in self.graph.graph.nodes():
        #     if isinstance(node, str):
        #         self.hubs.append(node)
        #     else:
        #         self.authorities.append(node)

        # hits algorithm
        for _ in range(num_iteration):
            new_a = {}
            new_h = {}

            for node in self.graph.graph.nodes():
                # print(f"Node is {node}")
                auth_score = 0.0
                hub_score = 0.0

                if node in self.authorities:
                    print(f"node is {node} in authorities")
                    predecessors = self.graph.get_predecessors(node)
                    for predecessor in predecessors:
                        auth_score += self.hubs.count(predecessor)
                else:
                    print(f"node is {node} in hubs")
                    successors = self.graph.get_successors(node)
                    for successor in successors:
                        hub_score += self.authorities.count(successor)

                new_a[node] = auth_score
                new_h[node] = hub_score

            self.authorities = list(new_a.keys())
            self.hubs = list(new_h.keys())

        # now let's sort them
        h_s = sorted(self.hubs, key=lambda x: self.hubs.count(x), reverse=True)[:max_result]
        a_s = sorted(self.authorities, key=lambda x: self.authorities.count(x), reverse=True)[:max_result]

        return a_s, h_s

    def hits(self, num_iteration=5, max_result=10):
        auth_scores = {node: 1.0 for node in self.authorities}
        hub_scores = {node: 1.0 for node in self.hubs}

        # hits algorithm
        for _ in range(num_iteration):
            new_auth_scores = {}
            new_hub_scores = {}

            # update authority score
            for node in auth_scores:
                new_auth_score = sum(hub_scores[pred] for pred in self.graph.get_predecessors(node))
                new_auth_scores[node] = new_auth_score

            # update hub score
            for node in hub_scores:
                new_hub_score = sum(auth_scores[succ] for succ in self.graph.get_successors(node))
                new_hub_scores[node] = new_hub_score

            # values could be as large as possible, let's normalize them
            auth_sum = sum(new_auth_scores.values())
            hub_sum = sum(new_hub_scores.values())
            auth_scores = {node: score / auth_sum for node, score in new_auth_scores.items()}
            hub_scores = {node: score / hub_sum for node, score in new_hub_scores.items()}

        h_s = sorted(self.hubs, key=lambda x: hub_scores.get(x, 0), reverse=True)[:max_result]
        a_s = sorted(self.authorities, key=lambda x: auth_scores.get(x, 0), reverse=True)[:max_result]

        return h_s, a_s


if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer

    # load corpus
    preprocessed_crawled_data_path = path_access.path_to_logic() + "IMDB_crawled_preprocessed.json"
    with open(preprocessed_crawled_data_path, 'r') as file:
        corpus = json.load(file)

    # root_set = []  # TODO: it should be a subset of your corpus
    root_set = random.sample(corpus, 50)
    root_set = corpus[:100]
    print("root set sampled")

    analyzer = LinkAnalyzer(root_set=root_set)

    analyzer.expand_graph(corpus=corpus)
    print("graph expanded")
    actors, movies = analyzer.hits(num_iteration=5, max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
