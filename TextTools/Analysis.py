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
from yellowbrick.text.freqdist import FreqDistVisualizer


from scipy import sparse
from scipy.cluster.hierarchy import dendrogram

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import TruncatedSVD, PCA, \
    LatentDirichletAllocation, NMF

from TextTools import Transformers


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


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import TruncatedSVD

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)
    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 10, shuffle=False)
    subset_fileids = next(loader.fileids(test=True))

    titles = list(corpus.title_tagged(fileids=subset_fileids))
    descriptions = list(corpus.description_tagged(fileids=subset_fileids))

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
    # plot_clusters
    # ==========================================================================
    if False:
        # decompose data to 2D
        reduce = Pipeline([
            ("norm", Transformers.TextNormalizer()),
            ("vect", Transformers.Text2OneHotVector()),
            ('pca', TruncatedSVD(n_components=2))
        ])

        X2d = reduce.fit_transform(titles)

        # plot Kmeans
        model = Pipeline([
            ("norm", Transformers.TextNormalizer()),
            ("vect", Transformers.Text2OneHotVector()),
            ('clusters', Transformers.KMeansClusters(k=3))
        ])

        clusters = model.fit_transform(titles)

        plot_clusters(X2d, clusters)
        # plot_clusters_2d

    # ==========================================================================
    # plot_clusters
    # ==========================================================================
    if False:
        prepare_data = Pipeline(
            [('normalize', Transformers.TextNormalizer()),
             ('vectorize', Transformers.Text2OneHotVector()
              )])

        cluster_data = Transformers.KMeansClusters(k=5)

        data = prepare_data.fit_transform(titles)
        labels = cluster_data.fit_transform((data))

        cluster_plotter = ClusterPlot2D(data, labels, method="NMF")
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
        data = prepare_data.fit_transform(titles)

        freq_plotter = TermFrequencyPlot(
            data,
            occurrence=False,
            n_terms=50
        )
        freq_plotter.plot()
    # --------------------------------------------------------------------------

