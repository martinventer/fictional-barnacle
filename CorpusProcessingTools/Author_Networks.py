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

from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, \
    BoxZoomTool, ResetTool, DataRange1d
from bokeh.models.graphs import from_networkx
from bokeh.palettes import Spectral4, Viridis256 , Category20

from tqdm import tqdm


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
        self.corpus = Elsevier_Corpus_Reader.ScopusRawCorpusReader(path)

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
        for article in tqdm(self.corpus.author_list(**kwargs),
                            ascii=True,
                            desc="building co-author network"):
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

    def co_author_network_bokeh_better(self, style="neighbourhood", **kwargs):

        # Prepare Data
        G = nx.Graph()
        G.add_nodes_from(list(set(self.corpus.author_name(**kwargs))))
        G.add_edges_from(self.build_co_author_network(**kwargs))

        # assign node alpha
        if True:
            node_alphas = {}
            node_alpha = 0.7
            for node in G:
                node_alphas[node] = node_alpha
            nx.set_node_attributes(G, node_alphas, "node_alphas")

        # assign node scale
        if True:
            node_sizes = {}
            node_scale = 30
            for node in G:
                node_sizes[node] = node_scale ** 1.1
            nx.set_node_attributes(G, node_sizes, "node_sizes")

        # assign node colour
        if True:
            communities_ = community.greedy_modularity_communities(G)
            if style is "max_clique":
                communities_ = nx.algorithms.clique.find_cliques(G)
            elif style is "neighbourhood":
                communities_ = \
                    nx.algorithms.components.connected_component_subgraphs(G)

            color_dic = {}
            for index, community_ in enumerate(communities_):
                for individual in community_:
                    color_dic[individual] = index

            node_colours = {}
            for node in G:
                node_colours[node] = Category20[20][color_dic[node] % 20]
            nx.set_node_attributes(G, node_colours, "node_colours")

        # assign edge colour
        edge_attrs = {}
        for start_node, end_node, _ in G.edges(data=True):
            edge_color = "black"
            edge_attrs[(start_node, end_node)] = edge_color

        nx.set_edge_attributes(G, edge_attrs, "edge_color")

        # Show with Bokeh
        range_scale = 25000
        plot = Plot(plot_width=1200, plot_height=1200,
                    x_range=Range1d(-(range_scale * 0.1), (range_scale * 1.1)),
                    y_range=Range1d(-(range_scale * 0.1), (range_scale * 1.1)))
        # plot = Plot(plot_width=800, plot_height=800,
        #             x_range=DataRange1d(),
        #             y_range=DataRange1d())
        # plot = Plot(plot_width=800, plot_height=800)
        plot.title.text = "Graph Interaction Demonstration"

        node_hover_tool = HoverTool(
            tooltips=[("index", "@index")])
        plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

        graph_renderer = from_networkx(G, graphviz_layout(G, prog="twopi"),
                                       scale=1,
                                       center=(0, 0))

        graph_renderer.node_renderer.glyph = Circle(size=15,
                                                    fill_alpha="node_alphas",
                                                    radius="node_sizes",
                                                    fill_color="node_colours")

        graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color",
                                                       line_alpha=0.8,
                                                       line_width=1)
        plot.renderers.append(graph_renderer)

        output_file("interactive_graphs.html")
        show(plot)


if __name__ == '__main__':
    AN = AuthorNetworks("Corpus/Processed_corpus/")
    # AN.plot_co_author_network(categories='soft robot/2000')
    # AN.plot_co_author_network()
    # AN.co_author_network_bokeh_better(categories='soft robot/2001')
    AN.co_author_network_bokeh_better()

