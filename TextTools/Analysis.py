#! envs/fictional-barnacle/bin/python3.6
"""
Analysis.py

@author: martinventer
@date: 2019-08-15

Tools for analysinig text data
"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from yellowbrick.text.freqdist import FreqDistVisualizer
import networkx as nx
from fuzzywuzzy import fuzz
from gensim.summarization.summarizer import summarize, summarize_corpus
from gensim.summarization import keywords
from gensim.corpora import Dictionary

from scipy import sparse
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
import itertools
import heapq
from operator import itemgetter
from tabulate import tabulate

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import TruncatedSVD, PCA, \
    LatentDirichletAllocation, NMF

import CorpusReaders.Corpus_filters
from TextTools import Transformers
from Utils.Utils import iter_flatten

from scipy.sparse import csr_matrix


def most_common_terms(docs, n_terms):
    """
    returns the most common terms in a set documents from a corpus.
    Parameters
    ----------
    Returns
    -------
        a tuple pair list
    """
    flat_observations = [i for i in iter_flatten(docs)]
    word_count = Counter(flat_observations)

    return zip(*word_count.most_common(n_terms))

class DendrogramPlot:
    """
    plotting class for Hierarchical clustering. builds a dendrogram from the
    children of a graph
    """

    def __init__(self, children) -> None:
        """
        initializes the children of the dendrogram and constructs the linkage
        matrix

        Parameters
        ----------
        children
        """
        self.children = children
        self.linkage_matrix = self.get_linkage()

    def get_linkage(self) -> np.array:
        """
        constructs the linkage matrix
        Returns
        -------

        """
        # determine distance between each pair
        distance = position = np.arange(self.children.shape[0])

        # create linkage matrix
        linkage_matrix = np.column_stack([
            self.children, distance, position
        ]).astype(float)

        return linkage_matrix

    def plot(self, **kwargs) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = dendrogram(self.linkage_matrix, **kwargs)
        plt.tick_params(axis='x', bottom='off', top='off', labelbottom="off")
        plt.tight_layout()
        plt.show()


class ClusterPlot2D:
    """
    plotting class for cluster results
    """
    methods = {"SVD" : TruncatedSVD,
               "PCA" : PCA,
               "LDA" : LatentDirichletAllocation,
               "NMF" : NMF,
               "SE" : SpectralEmbedding,
               "TSNE" : TSNE}

    def __init__(self, clusters, labels, method="TSNE"):
        self.clusters = clusters
        self.labels = labels
        self.decompose = ClusterPlot2D.methods[method](n_components=2)
        self.data_2d = None
        self.data_process()

    def data_process(self) -> None:
        """
        converts sparse matrix to dense 2D projection
        Returns
        -------
            None
        """
        if type(self.clusters) is sparse.csr.csr_matrix:
            self.clusters = self.clusters.toarray()
        X2d = self.decompose.fit_transform(self.clusters)
        x_min, x_max = np.min(X2d, axis=0), np.max(X2d, axis=0)
        self.data_2d = (X2d - x_min) / (x_max - x_min)

    def plot(self, **kwargs) -> None:
        """
        Plots a 2D scatter plot of the clusterd data with coloured text to
        labels for each data point.
        Parameters
        ----------
        kwargs
            additional plotting arguements

        Returns
        -------
            None
        """
        plt.figure(figsize=(15, 9))
        for i in range(self.data_2d.shape[0]):
            plt.text(self.data_2d[i, 0],
                     self.data_2d[i, 1],
                     str(self.labels[i]),
                     color=plt.cm.nipy_spectral(self.labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def simple_plot(self, **kwargs) -> None:
        """
        Plots a 2D scatter plot of the clusterd data with coloured points to
        labels for each data point.
        Parameters
        ----------
        kwargs :
            Additional plotting arguements

        Returns
        -------

        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.scatterplot(
            x=self.data_2d[:, 0],
            y=self.data_2d[:, 1],
            hue=self.labels,
            **kwargs)
        plt.tight_layout()
        plt.show()


class TermFrequencyPlot:
    """
    plotting term frequency given a list of terms per document. Works for
    terms, keyphrases and entities
    """

    def __init__(self,
                 docs,
                 occurrence=False,
                 features=None,
                 n_terms=100) -> None:
        """
        Initialize a term frequency plotter
        Parameters
        ----------
        occurrence : bool
            flag to switch output from term frequency in courpus to term
            occurrence per document in the corpus
        docs : List of lists
            a list of documents containing a list of terms per document
        n_terms : int
            the 'n' most common terms to be included in the plot
        """
        self.docs = docs
        self.features = features
        self.occurrence = occurrence
        self.n_terms = n_terms
        self.visualizer = None
        self.data_process()

    def data_process(self) -> None:
        """
        vectorizes the imput documents
        Returns
        -------
            None
        """
        if not self.features:
            if not self.occurrence:
                vectorizer = Transformers.Text2FrequencyVector()
                self.docs = vectorizer.fit_transform(self.docs)
                self.features = vectorizer.get_feature_names()
            else:
                vectorizer = Transformers.Text2OneHotVector()
                self.docs = vectorizer.fit_transform(self.docs)
                self.features = vectorizer.get_feature_names()

        self.visualizer = FreqDistVisualizer(features=self.features,
                                             n=self.n_terms)
        self.visualizer.fit(self.docs)

    def plot(self, **kwargs) -> None:
        """
        Plot the term frequency of a collection of documents
        Parameters
        ----------
        kwargs
            additional plotting arguements

        Returns
        -------
            None
        """
        self.visualizer.poof(**kwargs)


class TermCoocNetwork:
    """
    plots a network diagram of term co-occurance within a series of documents
    """

    def __init__(self, docs,
                 n_terms,
                 occurance_depth=2,
                 minimum_occurance=0) -> None:
        """

        Parameters
        ----------
        docs : List of lists
            a list of documents containing a list of terms per document
        n_terms : int
            the 'n' most common terms to be included in the plot
        minimum_occurance : int
            minimum number of occurance to register a connection.
        occurance_depth : int
            depth of the co-occurance evaluation. default is 2 so the occurance
            of a pair of terms will be tracked. If the value is increased a
            higher order occuance will be required.
        """
        self.docs = docs
        self.n_terms = n_terms
        self.occurance_depth = occurance_depth
        self.minimum_occurance = minimum_occurance
        self.g =None

    def cooc_dict(self, observations, terms) -> dict:
        """
        assembles a dictionary containing instances of co-occurance of terms in a
        list of listed observations
        Parameters
        ----------
        observations : corpus view
            document text data in the form [paragraph[sentance(token, tag)]]
        terms : list
            terms to consider in the co-occurance matrix
        Returns
        -------

        """
        # create a list of all possible paris of terms
        possible_pairs = list(itertools.combinations(terms,
                                                     self.occurance_depth))

        # initialise a dictionary containing an entry for each pair of words
        default_count = 0
        cooccurring = dict.fromkeys(possible_pairs, default_count)

        # incriment possible pairs for each occurance
        for observation in observations:
            for pair in possible_pairs:
                if pair[0] in observation and pair[1] in observation:
                    cooccurring[pair] += 1

        # remove cases where the co-occurance is lest than a set mimimum
        cooccurring = {k: v for k, v in cooccurring.items() if
                       v > self.minimum_occurance}

        return cooccurring

    def plot_pos(self, method="kamada") -> None:
        if method is "kamada":
            self.pos = nx.kamada_kawai_layout(self.g)
        elif method is "dot":
            self.pos = nx.graphviz_layout(self.g, prog="dot")
        elif method is "neato":
            self.pos = nx.graphviz_layout(self.g, prog="neato")
        elif method is "fdp":
            self.pos = nx.graphviz_layout(self.g, prog="fdp")
        elif method is "sfdp":
            self.pos = nx.graphviz_layout(self.g, prog="sfdp")
        elif method is "twopi":
            self.pos = nx.graphviz_layout(self.g, prog="twopi")
        elif method is "circo":
            self.pos = nx.graphviz_layout(self.g, prog="circo")
        elif method is "spring":
            self.pos = nx.spring_layout(self.g)
        elif method is "spectral":
            self.pos = nx.spectral_layout(self.g)

    def create_network(self, method="kamada") -> None:
        self.g = nx.Graph()
        # create nodes for each term
        frequent_terms, term_count = most_common_terms(self.docs, self.n_terms)
        term_counts = dict(zip(frequent_terms, term_count))
        self.g.add_nodes_from(list(set(frequent_terms)))

        # create edges for each pair of connected terms
        pairs = self.cooc_dict(observations=self.docs,
                               terms=frequent_terms)

        self.edge_width = []
        edge_scale = 0.1
        for pair, wgt in pairs.items():
            self.g.add_edge(pair[0], pair[1])
            self.edge_width.append(edge_scale * wgt)

        # set layout style
        self.plot_pos(method=method)

        self.node_options = {'alpha': 0.5}
        # assign node scale to a multiple of its term count.
        if True:
            node_size = []
            node_scale = 10.0
            for node in self.g:
                node_size.append(node_scale * term_counts[node])
            self.node_options["node_size"] = node_size

    def plot(self, term=None) -> None:
        if not term:
            graph = self.g
        else:
            graph = nx.ego_graph(self.g, term)

        fig, ax = plt.subplots(1, figsize=(15, 12))
        # draw the nodes
        nx.draw_networkx_nodes(graph, pos=self.pos, **self.node_options)

        if True:
            node_names = {}
            for node in graph:
                node_names[node] = node  # .split()[-1]
            nx.draw_networkx_labels(graph, self.pos,
                                    labels=node_names,
                                    font_size=8)

        # draw in the edges
        nx.draw_networkx_edges(graph, pos=self.pos,
                               alpha=0.2,
                               width=self.edge_width)

        # plot the figure
        ax.axis('off')
        fig.tight_layout()
        plt.show()

    def info(self, term=None) -> None:
        if not term:
            graph = self.g
        else:
            graph = nx.ego_graph(self.g, term)
        print(nx.info(graph))

    def nbest_centrality(self, graph, metrics, n=10):
        # computer centrallity score
        nbest = {}
        for name, metric in metrics.items():
            scores = metric(graph)

            # ser the score as a property oon each node
            nx.set_node_attributes(graph, name=name, values=scores)

            # Find the top n scores and print them along with their index
            topn = heapq.nlargest(n, scores.items(), key=itemgetter(1))
            nbest[name] = topn

        return nbest

    def print_centralities(self, term=None):
        if not term:
            graph = self.g
        else:
            graph = nx.ego_graph(self.g, term)

        centralities = {"Degree Centrality": nx.degree_centrality,
                        "Betweenness Centrality": nx.betweenness_centrality,
                        "Closeness": nx.closeness_centrality,
                        "eigenvector": nx.eigenvector_centrality,
                        "katz": nx.katz_centrality_numpy,
                        "pagerank": nx.pagerank_numpy}

        centrality = self.nbest_centrality(graph, centralities, 10)

        for measure, scores in centrality.items():
            print("Ranks for {}:".format(measure))
            print(tabulate(scores,headers=["Top Terms", "Socre"]))
            print("")

    def plot_distributions(self, term=None):
        if not term:
            graph = self.g
        else:
            graph = nx.ego_graph(self.g, term)

        sns.distplot([graph.degree(v) for v in graph.nodes()], norm_hist=True)
        plt.show()

    def print_structure(self, term=None):
        if not term:
            graph = self.g
        else:
            graph = nx.ego_graph(self.g, term)

        print("Average clustering coefficient: {}".format(
            nx.average_clustering(graph)
        ))
        print("Transistivity: {}".format(
            nx.transitivity(graph)
        ))
        print("Number of cliques: {}".format(
            nx.graph_number_of_cliques(graph)
        ))

    def pairwise_comparisons(self, graph):
        return itertools.combinations(graph.nodes(), 2)

    def edge_blocked_comparisons(self, graph):
        for n1, n2 in self.pairwise_comparisons(graph):
            hood1 = frozenset(graph.neighbors(n1))
            hood2 = frozenset(graph.neighbors(n2))
            if hood1 & hood2:
                yield n1, n2

    def similarity(self, G,  n1, n2):
        """
        Returns the mean of the partial_ratio score for each field in the two
        entities. Note that if they don't have fields that match, the score will
        be zero.
        """
        scores = [
            fuzz.partial_ratio(n1, n2),
            fuzz.partial_ratio(G.node[n1]['type'], G.node[n2]['type'])
        ]

        return float(sum(s for s in scores)) / float(len(scores))

    def fuzzy_blocked_comparisons(self, G, threshold=65):
        """
        A generator of pairwise comparisons, that highlights comparisons between
        nodes that have an edge to the same entity, but filters out comparisons
        if the similarity of n1 and n2 is below the threshold.
        """
        for n1, n2 in self.pairwise_comparisons(G):
            hood1 = frozenset(G.neighbors(n1))
            hood2 = frozenset(G.neighbors(n2))
            if hood1 & hood2:
                if self.similarity(G, n1, n2) > threshold:
                    yield n1, n2

    def info_extend(self, term=None):
        """
        Wrapper for nx.info with some other helpers.
        """
        if not term:
            graph = self.g
        else:
            graph = nx.ego_graph(self.g, term)

        pairwise = len(list(self.pairwise_comparisons(graph)))
        edge_blocked = len(list(self.edge_blocked_comparisons(graph)))
        # fuzz_blocked = len(list(self.fuzzy_blocked_comparisons(graph)))

        output = [""]
        output.append("Number of Pairwise Comparisons: {} ".format(pairwise))
        output.append(
            "Number of Edge Blocked Comparisons: {}".format(edge_blocked))
        # output.append(
        #     "Number of Fuzzy Blocked Comparisons: {}".format(fuzz_blocked))

        print(output)


class TermCoocMatrix:
    """
    plots a Cooc matrix of term co-occurance within a series of documents
    """

    def __init__(self, docs,
                 n_terms) -> None:
        """

        Parameters
        ----------
        docs : List of lists
            a list of documents containing a list of terms per document
        n_terms : int
            the 'n' most common terms to be included in the plot

        """
        self.docs = docs
        self.n_terms = n_terms

    def cooc_matrix(self, allowed_words):
        """
        creates a co-occurance matrix given a list off search terms and documents
        Parameters
        ----------
        allowed_words : List
            list of terms to find

        Returns
        -------
            A fully dense numpy array for the co-occurance matrix
            a list of word ids

        """
        # make sure that the input text is in a flat list.
        allowed_words = [i for i in iter_flatten(allowed_words)]

        # convert the word list to a distionary of terms with unique intiger ids
        word_to_id = dict(zip(allowed_words, list(range(len(allowed_words)))))

        # create a sorted document  lookup
        documents_as_ids = [
            np.sort([word_to_id[w]
                     for w in doc if w in word_to_id]).astype('uint32')
            for doc in self.docs]

        # Create a 2D array of terms vs terms
        row_ind, col_ind = zip(
            *itertools.chain(*[[(i, w) for w in doc]
                               for i, doc in enumerate(documents_as_ids)]))

        # Generate an array of term counts per document
        # use unsigned int for better memory utilization
        data = np.ones(len(row_ind), dtype='uint32')
        max_word_id = max(itertools.chain(*documents_as_ids)) + 1
        docs_words_matrix = csr_matrix((data,
                                        (row_ind, col_ind)),
                                       shape=(len(documents_as_ids),
                                              max_word_id))

        # multiplying docs_words_matrix with its transpose matrix to
        # generate the co-occurences matrix
        words_cooc_matrix = docs_words_matrix.T * docs_words_matrix
        words_cooc_matrix.setdiag(0)
        return words_cooc_matrix.toarray(), word_to_id

    def create_matrix(self) -> None:
        # create nodes for each term
        frequent_terms, term_count = most_common_terms(self.docs, self.n_terms)

        # By frequency
        mtx, ids = self.cooc_matrix(frequent_terms)
        self.mtx = mtx
        self.frequent_terms = frequent_terms

    def plot(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 6))

        x_tick_marks = np.arange(self.n_terms)
        y_tick_marks = np.arange(self.n_terms)

        ax.set_xticks(x_tick_marks)
        ax.set_yticks(y_tick_marks)
        ax.set_xticklabels(self.frequent_terms, fontsize=8, rotation=90)
        ax.set_yticklabels(self.frequent_terms, fontsize=8)
        ax.xaxis.tick_top()
        ax.set_xlabel("By Frequency")
        plt.imshow(self.mtx.tolist(), norm=LogNorm(), interpolation='nearest',
                   cmap='YlOrBr')
        plt.show()


class TermTemporal:
    """
    plots the occurance over time
    """

    def __init__(self, docs,
                 dates,
                 n_terms=50) -> None:
        """

        Parameters
        ----------
        docs : List of lists
            a list of documents containing a list of terms per document
        n_terms : int
            the 'n' most common terms to be included in the plot

        """
        self.docs = docs
        self.dates = dates
        self.n_terms = n_terms

    def create_connections(self):
        # extract the most common terms in the documnet list
        frequent_terms, term_count = most_common_terms(self.docs, self.n_terms)

        x, y = [], []
        for doc, date in zip(self.docs, self.dates):
            for i, term in enumerate(frequent_terms):
                if term in doc:
                    x.append(date)
                    y.append(i)

        self.x = x
        self.y = y
        self.frequent_terms = frequent_terms

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        plt.plot(self.x, self.y, "*")

        plt.yticks(list(range(len(self.frequent_terms))), self.frequent_terms,
                   size=8)
        plt.ylim(-1, len(self.frequent_terms))
        plt.title("Occurance of the {0} most frequent terms".format(
            self.n_terms))
        plt.show()


class Summary:
    def __init__(self):
        pass

    def text_summary(self, text):
        return summarize(text, ratio=0.01)

    def text_keywords(self, text):
        return keywords(text)

    def corpus_summary(self, docs):
        tokens = [str(i) for i in iter_flatten(data)]
        dictionary = Dictionary(docs)
        corpus = [dictionary.doc2bow(doc) for doc in
                  docs]
        selected_docs = summarize_corpus(corpus, ratio=0.001)
        sumsum = []
        for doc_number, document in enumerate(selected_docs):
            # Retrieves all words from the document.
            words = [dictionary[token_id] for (token_id, count) in document]
            sumsum.append(words)
        return selected_docs


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import TruncatedSVD

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)
    loader = CorpusReaders.Corpus_filters.CorpuKfoldLoader(corpus, 10, shuffle=False)
    subset_fileids = next(loader.fileids(test=True))

    # ==========================================================================
    # DendrogramPlot
    # ==========================================================================
    if False:
        model = Pipeline([
            ("norm", Transformers.TextNormalizer()),
            ("vect", Transformers.Text2OneHotVector()),
            ('clusters', Transformers.HierarchicalClustering())
        ])

        clusters = model.fit_transform(titles)
        children = model.named_steps['clusters'].children

        tree_plotter = DendrogramPlot(children)
        tree_plotter.plot()

    # ==========================================================================
    # ClusterPlot2D
    # ==========================================================================
    if False:
        prepare_data = Pipeline(
            [('normalize', Transformers.TextNormalizer()),
             ('vectorize', Transformers.Text2OneHotVector()
              )])

        cluster_data = Transformers.KMeansClusters(k=5)

        data = prepare_data.fit_transform(titles)
        labels = cluster_data.fit_transform((data))

        cluster_plotter = ClusterPlot2D(data, labels, method="SE")
        cluster_plotter.plot()
        cluster_plotter.simple_plot()

    # ==========================================================================
    # TermFrequencyPlot
    # ==========================================================================
    if False:
        prepare_data = Pipeline(
            [('normalize', Transformers.TextNormalizer())
             ])

        titles = list(corpus.title_tagged(fileids=subset_fileids))
        descriptions = list(corpus.description_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(descriptions)

        freq_plotter = TermFrequencyPlot(
            data,
            occurrence=False,
            n_terms=100
        )
        freq_plotter.plot()
    # --------------------------------------------------------------------------
    if False:
        prepare_data = Pipeline(
            [('phrases', Transformers.KeyphraseExtractorS())
             ])

        titles = list(corpus.title_tagged(fileids=subset_fileids))
        descriptions = list(corpus.description_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(titles)

        freq_plotter = TermFrequencyPlot(
            data,
            occurrence=False,
            n_terms=100
        )
        freq_plotter.plot()
    # --------------------------------------------------------------------------
    if False:
        prepare_data = Pipeline(
            [('phrases', Transformers.KeyphraseExtractorL())
             ])

        titles = list(corpus.title_tagged(fileids=subset_fileids))
        descriptions = list(corpus.description_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(titles)

        freq_plotter = TermFrequencyPlot(
            data,
            occurrence=False,
            n_terms=100
        )
        freq_plotter.plot()
    # --------------------------------------------------------------------------
    if False:
        prepare_data = Pipeline(
            [('entities', Transformers.EntityExtractor())
             ])

        titles = list(corpus.title_tagged(fileids=subset_fileids))
        descriptions = list(corpus.description_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(descriptions)

        freq_plotter = TermFrequencyPlot(
            data,
            occurrence=False,
            n_terms=50
        )
        freq_plotter.plot()

    # ==========================================================================
    # TermCoocNetwork
    # ==========================================================================
    if False:
        prepare_data = Pipeline(
            [('normalize', Transformers.TextNormalizer())
             ])
        # titles = list(corpus.title_tagged(fileids=subset_fileids))
        titles = list(corpus.title_tagged())
        data = prepare_data.fit_transform(titles)

        network_plotter = TermCoocNetwork(data, n_terms=50)
        network_plotter.create_network()
        # network_plotter.info()
        # network_plotter.info("hand")
        # network_plotter.print_centralities()
        # network_plotter.print_centralities("hand")
        network_plotter.plot()
        # network_plotter.plot("hand")
        # network_plotter.plot_distributions()
        # network_plotter.plot_distributions("hand")
        # network_plotter.print_structure()
        # network_plotter.print_structure("hand")
        # network_plotter.edge_blocked_comparison()
        # network_plotter.info_extend()
    # --------------------------------------------------------------------------
    if False:
        prepare_data = Pipeline(
            [('normalize', Transformers.TextNormalizer())
             ])

        descriptions = list(corpus.description_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(descriptions)

        network_plotter = TermCoocNetwork(data, n_terms=100)
        network_plotter.create_network()
        # network_plotter.info()
        # network_plotter.print_centralities()
        # network_plotter.plot()
        # network_plotter.plot_ego("2")
        # network_plotter.plot_distributions()
    # --------------------------------------------------------------------------
    if False:
        prepare_data = Pipeline(
            [('entities', Transformers.EntityExtractor())
             ])

        titles = list(corpus.title_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(titles)

        network_plotter = TermCoocNetwork(data, n_terms=100)
        network_plotter.create_network(method='spring')
        network_plotter.details()
        network_plotter.plot()
    # --------------------------------------------------------------------------
    if False:
        prepare_data = Pipeline(
            [('entities', Transformers.EntityExtractor())
             ])

        descriptions = list(corpus.description_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(descriptions)

        network_plotter = TermCoocNetwork(data, n_terms=100)
        network_plotter.create_network(method='spring')
        network_plotter.details()
        network_plotter.plot()
    # --------------------------------------------------------------------------
    if False:
        prepare_data = Pipeline(
            [('entities', Transformers.KeyphraseExtractorL())
             ])

        titles = list(corpus.title_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(titles)

        network_plotter = TermCoocNetwork(data, n_terms=100)
        network_plotter.create_network(method='spring')
        network_plotter.details()
        network_plotter.plot()
    # --------------------------------------------------------------------------
    if False:
        prepare_data = Pipeline(
            [('entities', Transformers.KeyphraseExtractorL())
             ])

        descriptions = list(corpus.description_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(descriptions)

        network_plotter = TermCoocNetwork(data, n_terms=100)
        network_plotter.create_network(method='spring')
        network_plotter.details()
        network_plotter.plot()
    # ==========================================================================
    # TermCoocMatrix
    # ==========================================================================
    if False:
        prepare_data = Pipeline(
            [('normalize', Transformers.TextNormalizer())
             ])
        titles = list(corpus.title_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(titles)

        network_plotter = TermCoocMatrix(data, n_terms=50)
        network_plotter.create_matrix()
        network_plotter.plot()
    # --------------------------------------------------------------------------
    if False:
        prepare_data = Pipeline(
            [('normalize', Transformers.EntityExtractor())
             ])
        titles = list(corpus.title_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(titles)

        network_plotter = TermCoocMatrix(data, n_terms=50)
        network_plotter.create_matrix()
        network_plotter.plot()
    # ==========================================================================
    # TermTemporal
    # ==========================================================================
    if False:
        prepare_data = Pipeline(
            [('normalize', Transformers.TextNormalizer())
             ])
        titles = list(corpus.title_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(titles)

        network_plotter = TermTemporal(
            data,
            corpus.publication_date(fileids=subset_fileids),
            n_terms=50)
        network_plotter.create_connections()
        network_plotter.plot()
    # ==========================================================================
    # Summary
    # ==========================================================================
    if True:
        prepare_data = Pipeline(
            # [('normalize', Transformers.TextNormalizer())
            [('normalize', Transformers.TextSimpleTokenizer())
             ])
        titles = list(corpus.title_tagged(fileids=subset_fileids))
        descriptions = list(corpus.description_tagged(fileids=subset_fileids))
        data = prepare_data.fit_transform(descriptions)

        # summary = Summary()
        # print(summary.corpus_summary(data))

        # tokens = [str(i) for i in iter_flatten(data)]
        dictionary = Dictionary(data)
        corpus = [dictionary.doc2bow(doc) for doc in
                  data]
        selected_docs = summarize_corpus(corpus, ratio=0.01)
        sumsum = []
        for doc_number, document in enumerate(selected_docs):
            # Retrieves all words from the document.
            # words = [dictionary[token_id] for (token_id, count) in document]
            # sumsum.append(words)
            words = [dictionary[token_id] for (token_id, count) in document]
            print(" ".join(words) + "\n \n")

    # --------------------------------------------------------------------------

