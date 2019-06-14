# /home/martin/Documents/RESEARCH/fictional-barnacle/CorpusProcessingTools/
"""
Author_Networks.py

@author: martinventer
@date: 2019-06-14

Reads the pre-processed Corpus data and generates bibliometric plots
"""

from CorpusReader import Elsevier_Corpus_Reader

from itertools import combinations
from collections import Counter

import networkx as nx
from networkx.algorithms import community
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


class AuthorNetworks():
    """
    Build and plot an author network
    """
    def __init__(self, path):
        """
        Initialise the author network
        Parameters
        ----------
        path : string like
            path to corpus

        """
        self.path = path
        self.corpus = Elsevier_Corpus_Reader.ScopusPickledCorpusReader(path)

    def build_co_author_network(self, **kwargs) -> list:
        """
        build a dictionary with authors as keys, and a list  of collaborating
        authors
        Parameters
        ----------

        Returns
        -------
            dict

        """

        collaborations = []
        for article in self.corpus.author_list(**kwargs):
            collaborations += list(combinations(article, 2))

        return collaborations

    def plot_co_author_network(self, style="neighbourhood", **kwargs):
        """
        plot a simple co-author network
        Parameters
        ----------
        style : str default "neighbourhood"
            "neighbourhood" -
            "max_clique" -
        kwargs

        Returns
        -------

        """
        fig, ax = plt.subplots(1, figsize=(15, 12))
        G = nx.Graph()
        G.add_nodes_from(list(set(self.corpus.author_name(**kwargs))))
        G.add_edges_from(self.build_co_author_network(**kwargs))

        node_options = {'alpha': 0.5}

        # assign node scale
        if True:
            node_size = []
            node_scale = 100
            for node in G:
                node_size.append(node_scale ** 1.1)
            node_options["node_size"] = node_size

        # assign node colour
        if True:
            color_map = []
            communities_ = community.greedy_modularity_communities(G)
            if style is "max_clique":
                communities_ = nx.algorithms.clique.find_cliques(G)
            elif style is "neighbourhood":
                communities_ = \
                    nx.algorithms.components.connected_component_subgraphs(G)

            color_dic = {}
            # com_counter = 0
            for index, community_ in enumerate(communities_):
                for individual in community_:
                    color_dic[individual] = index
                # com_counter += 1

            for node in G.nodes:
                color_map.append(color_dic[node])
            node_options["node_color"] = color_map
            node_options["cmap"] = plt.cm.get_cmap('rainbow')

        # set edge style
        edge_width = []
        edge_scale = 2
        for u, v, d in G.edges(data=True):
            edge_width.append(edge_scale ** 1.1)

        # adjust the node layout style
        pos = graphviz_layout(G, prog="twopi")
        nx.draw_networkx_nodes(G, pos=pos, **node_options)

        # add in the node lables
        if True:
            node_names = {}
            for node in G:
                node_names[node] = node#.split()[-1]
            nx.draw_networkx_labels(G, pos,
                                    labels=node_names,
                                    font_size=8)

        # draw in the edges
        nx.draw_networkx_edges(G, pos=pos,
                               width=edge_width,
                               alpha=0.2)

        # plot the figure
        ax.axis('off')
        fig.tight_layout()
        plt.show()
        

if __name__ == '__main__':
    AN = AuthorNetworks("Corpus/Processed_corpus/")
    # temp = AN.build_co_author_network(categories='soft robot/2000')
    AN.plot_co_author_network(categories='soft robot/2000')

