#! envs/fictional-barnacle/bin/python3.6
"""
Cluster_Plotting.py

@author: martinventer
@date: 2019-07-03

Tools for plotting clusters
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from sklearn.pipeline import Pipeline


def plot_dendrogram(children, **kwargs):
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




if __name__ == '__main__':
    from CorpusReader import Elsevier_Corpus_Reader
    from CorpusProcessingTools import Corpus_Vectorizer
    from CorpusProcessingTools import Corpus_Cluster

    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 100, shuffle=False)
    subset = next(loader.fileids(test=True))

    docs = list(corpus.title_tagged(fileids=subset))
    pickles = subset

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
