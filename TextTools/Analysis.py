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
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.manifold import TSNE, SpectralEmbedding

from sklearn.decomposition import TruncatedSVD, PCA, \
    LatentDirichletAllocation, NMF


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


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader
    from TextTools import Transformers
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import TruncatedSVD

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)
    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 10, shuffle=False)
    subset_fileids = next(loader.fileids(test=True))

    observations = list(corpus.title_tagged(fileids=subset_fileids))

    # ==========================================================================
    # DendrogramPlot
    # ==========================================================================
    if False:
        model = Pipeline([
            ("norm", Transformers.TextNormalizer()),
            ("vect", Transformers.Text2OneHotVector()),
            ('clusters', Transformers.HierarchicalClustering())
        ])

        clusters = model.fit_transform(observations)
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

        X2d = reduce.fit_transform(observations)

        # plot Kmeans
        model = Pipeline([
            ("norm", Transformers.TextNormalizer()),
            ("vect", Transformers.Text2OneHotVector()),
            ('clusters', Transformers.KMeansClusters(k=3))
        ])

        clusters = model.fit_transform(observations)

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

        data = prepare_data.fit_transform(observations)
        labels = cluster_data.fit_transform((data))

        cluster_plotter = ClusterPlot2D(data, labels, method="NMF")
        cluster_plotter.plot()
        cluster_plotter.simple_plot()
