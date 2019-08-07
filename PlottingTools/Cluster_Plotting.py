#! envs/fictional-barnacle/bin/python3.6
"""
Cluster_Plotting.py

@author: martinventer
@date: 2019-07-03

Tools for plotting clusters
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram

from sklearn.manifold import SpectralEmbedding, TSNE

from sklearn.decomposition import PCA, LatentDirichletAllocation, \
    TruncatedSVD, NMF


from sklearn.pipeline import Pipeline


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


    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax = sns.scatterplot(x=X2d[:, 0], y=X2d[:, 1], hue=y)
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader
    from CorpusProcessingTools import Corpus_Vectorizer
    from CorpusProcessingTools import Corpus_Cluster

    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 100, shuffle=False)
    subset = next(loader.fileids(test=True))

    docs = list(corpus.title_tagged(fileids=subset))

    # --------------------------------------------------------------------------
    # plot_dendrogram
    # --------------------------------------------------------------------------
    # model = Pipeline([
    #     ("norm", Corpus_Vectorizer.TitleNormalizer()),
    #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
    #     ('clusters', Corpus_Cluster.HierarchicalClustering())
    # ])
    #
    # clusters = model.fit_transform(observations)
    # labels = model.named_steps['clusters'].labels
    # children = model.named_steps['clusters'].children
    #
    # plot_dendrogram(children)

    # --------------------------------------------------------------------------
    # plot_clusters
    # --------------------------------------------------------------------------
    # decompose data to 2D
    reduce = Pipeline([
        ("norm", Corpus_Vectorizer.TitleNormalizer()),
        ("vect", Corpus_Vectorizer.OneHotVectorizer()),
        ('pca', PCA(n_components=2))
    ])

    X2d = reduce.fit_transform(docs)

    # plot Kmeans
    model = Pipeline([
        ("norm", Corpus_Vectorizer.TitleNormalizer()),
        ("vect", Corpus_Vectorizer.OneHotVectorizer()),
        ('clusters', Corpus_Cluster.MiniBatchKMeansClusters(k=3))
    ])

    clusters = model.fit_transform(docs)

    plot_clusters(X2d, clusters)


