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
from scipy.cluster.hierarchy import dendrogram

from sklearn.decomposition import PCA

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
    # fig, ax = plt.subplot()
    ax = dendrogram(linkage_matrix, **kwargs)
    plt.tick_params(axis='x', bottom='off', top='off', labelbottom="off")
    plt.tight_layout()
    plt.show()


def plot_clusters(X, y, **kwargs) -> None:

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.scatterplot(x=X[:,0], y=X[:,1], hue=y)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader
    from CorpusProcessingTools import Corpus_Vectorizer
    from CorpusProcessingTools import Corpus_Cluster


    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 100, shuffle=False)
    subset = next(loader.fileids(test=True))

    docs = list(corpus.title_tagged(fileids=subset))

    # # Plot hierarchical clustering
    # model = Pipeline([
    #     ("norm", Corpus_Vectorizer.TitleNormalizer()),
    #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
    #     ('clusters', Corpus_Cluster.HierarchicalClustering())
    # ])
    #
    # clusters = model.fit_transform(docs)
    # labels = model.named_steps['clusters'].labels
    # children = model.named_steps['clusters'].children
    #
    # plot_dendrogram(children)

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


