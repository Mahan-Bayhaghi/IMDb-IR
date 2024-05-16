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
        self.hubs = []
        self.authorities = []
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
                    print(f"star is {star}")
                    self.graph.add_node(star)
                    self.graph.add_edge(star, movie_id)
        # nx.draw(self.graph)
        # print(f"{self.graph.get_predecessors(ido)}")
        print("done")

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
        for movie in corpus:
            # TODO
            pass

    def hits(self, num_iteration=5, max_result=10):
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

        return a_s, h_s


if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer

    # load corpus
    preprocessed_crawled_data_path = path_access.path_to_logic() + "IMDB_crawled_preprocessed.json"
    with open(preprocessed_crawled_data_path, 'r') as file:
        corpus = json.load(file)

    # root_set = []  # TODO: it should be a subset of your corpus
    root_set = random.sample(corpus, int(len(corpus)/100))

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
