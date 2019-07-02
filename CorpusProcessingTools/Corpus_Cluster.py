#! envs/fictional-barnacle/bin/python3.6
"""
Corpus_Vectorizer.py

@author: martinventer
@date: 2019-06-28

Tools Clustering corpus
"""

import nltk

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from nltk.cluster import KMeansClusterer
from sklearn.cluster import MiniBatchKMeans

from sklearn.cluster import AgglomerativeClustering


class KMeansClusters(BaseEstimator, TransformerMixin):
    """
    Cluster text data using k-means. Makes use of nltk k-means clustering.
    Allows for alternative distance measures
    """
    def __init__(self, k=7):
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(self.k, self.distance,
                                     avoid_empty_clusters=True)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        """
        fits the K-means model to the given documents
        Parameters
        ----------
        documents :
            a string containing the normalized text.

        Returns
        -------
            fitted model
        """
        return self.model.cluster(documents, assign_clusters=True)


class MiniBatchKMeansClusters(BaseEstimator, TransformerMixin):
    """
    Cluster text data using k-means, in minibatch mode. Only uses euclidean
    distance

    """
    def __init__(self, k=7):
        self.k = k
        self.model = MiniBatchKMeans(self.k)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        """
        fits the K-means model to the given documents
        Parameters
        ----------
        documents :
            a string containing the normalized text.

        Returns
        -------
            fitted model
        """
        return self.model.fit_predict(documents)


class HierarchicalClustering(object):

    def __init__(self):
        self.model = AgglomerativeClustering()

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        """
        fits an agglomerative clustering to given vector
        Parameters
        ----------
        documents :
            a string containing the normalized text.

        Returns
        -------
            fitted model

        """
        clusters = self.model.fit_predict(documents)
        self.labels = clusters.labels_
        self.children = clusters.children_

        return clusters


if __name__ == '__main__':
    from CorpusReader import Elsevier_Corpus_Reader
    from CorpusProcessingTools import Corpus_Vectorizer

    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    loader = Elsevier_Corpus_Reader.CorpusLoader(corpus, 12, shuffle=False)

    docs = list(corpus.title_tagged(fileids=loader.fileids(1, test=True)))
    pickles = list(loader.fileids(1, test=True))

    # # K-means clustering pipeline
    # model = Pipeline([
    #     ("norm", Corpus_Vectorizer.TitleNormalizer()),
    #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
    #     # ('clusters', KMeansClusters(k=7)) # uses nltk k-means, allows
    #     different measures of distance
    #     ('clusters', MiniBatchKMeansClusters(k=7)) # uses sklearn
    #     clustering with minibatch, but no choice of distance measures
    # ])
    #
    # clusters = model.fit_transform(docs)
    #
    # for idx, cluster in enumerate(clusters):
    #     print("Document '{}' assigned to cluster {}.".format(pickles[idx],
    #                                                          cluster))

    # Agglomerative hierarchical clustering pipeline
    model = Pipeline([
        ("norm", Corpus_Vectorizer.TitleNormalizer()),
        ("vect", Corpus_Vectorizer.OneHotVectorizer()),
        ('clusters', HierarchicalClustering())
    ])

    model.fit_transform(docs)
    labels = model.named_steps['clusters'].labels

    #
    # for idx, fileid in enumerate(pickles):
    #     print("Document '{}' assigned to cluster {}.".format(fileid,
    #                                                          labels[idx]))
