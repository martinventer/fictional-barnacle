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
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, TransformerMixin


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




def plot_dendrogram(children, **kwargs) -> None:
    # determine distance between each pair
    distance = position = np.arange(children.shape[0])

    # create linkage matrix
    linkage_matrix = np.column_stack([
        children, distance, position
    ]).astype(float)

    # plot dendrogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = dendrogram(linkage_matrix, **kwargs)
    plt.tick_params(axis='x', bottom='off', top='off', labelbottom="off")
    plt.tight_layout()
    plt.show()


def plot_clusters(X, y, **kwargs) -> None:

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.scatterplot(x=X[:,0], y=X[:,1], hue=y)
    plt.tight_layout()
    plt.show()


def plot_clusters_2d(X, y, **kwargs) -> None:
    if type(X) is sparse.csr.csr_matrix:
        X = X.toarray()
    # reduce = TruncatedSVD(n_components=2)
    # reduce = PCA(n_components=2)
    # reduce = LatentDirichletAllocation(n_components=2)
    # reduce = NMF(n_components=2)
    # reduce = SpectralEmbedding(n_components=2)
    reduce = TSNE(n_components=2)

    X2d = reduce.fit_transform(X)

    x_min, x_max = np.min(X2d, axis=0), np.max(X2d, axis=0)
    X_red = (X2d - x_min) / (x_max - x_min)

    plt.figure(figsize=(15, 9))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader
    from TextTools import Transformers
    from sklearn.pipeline import Pipeline

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)
    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 10, shuffle=False)
    subset_fileids = next(loader.fileids(test=True))

    observations = list(corpus.title_tagged(fileids=subset_fileids))

    # ==========================================================================
    # DendrogramPlot
    # ==========================================================================
    model = Pipeline([
        ("norm", Transformers.TextNormalizer()),
        ("vect", Transformers.Text2OneHotVector()),
        ('clusters', Transformers.HierarchicalClustering())
    ])

    clusters = model.fit_transform(observations)
    children = model.named_steps['clusters'].children


    tree_plotter = DendrogramPlot(children)
    tree_plotter.plot()
