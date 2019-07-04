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

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

from CorpusProcessingTools import Corpus_Vectorizer

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
        return np.array(self.model.cluster(documents, assign_clusters=True))


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
        # clusters = self.model.fit_predict(documents)
        # self.labels = self.model.labels_
        self.labels = self.model.fit_predict(documents)
        self.children = self.model.children_

        return self.labels


# def identity(words):
#     return words


class SklearnTopicModels(object):

    def __init__(self, n_components=50, estimator='LDA'):
        """
        n_topics is the desired number of topics
        To use Latent Semantic Analysis, set estimator to 'LSA',
        To use Non-Negative Matrix Factorization, set estimator to 'NMF',
        otherwise, defaults to Latent Dirichlet Allocation ('LDA').
        """
        self.n_components = n_components

        if estimator == 'LSA':
            self.estimator = TruncatedSVD(n_components=self.n_components)
        elif estimator == 'NMF':
            self.estimator = NMF(n_components=self.n_components)
        else:
            self.estimator = LatentDirichletAllocation(
                n_components=self.n_components)

        self.model = Pipeline([
            ('norm', Corpus_Vectorizer.TextNormalizer()),
            # ('vect', Corpus_Vectorizer.CorpusFrequencyVector()),
            # ('vect', Corpus_Vectorizer.CorpusOneHotVector()),
            ('vect', Corpus_Vectorizer.CorpusTFIDVector()),
            ('model', self.estimator)
        ])

    def fit_transform(self, documents):
        self.model.fit_transform(documents)

        return self.model

    def get_topics(self, n=25):
        """
        n is the number of top terms to show for each topic
        """
        vectorizer = self.model.named_steps['vect']
        model = self.model.steps[-1][1]
        names = vectorizer.get_feature_names()
        topics = dict()

        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[:-(n - 1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens

        return topics


if __name__ == '__main__':
    from CorpusReader import Elsevier_Corpus_Reader
    from CorpusProcessingTools import Corpus_Vectorizer

    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 12, shuffle=False)
    subset = next(loader.fileids(test=True))

    docs = list(corpus.title_tagged(fileids=subset))
    pickles = subset

    # # K-means clustering pipeline
    # model = Pipeline([
    #     ("norm", Corpus_Vectorizer.TitleNormalizer()),
    #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
    #     ('clusters', KMeansClusters(k=7)) # uses nltk k-means, allows
    #     # different measures of distance
    #     # ('clusters', MiniBatchKMeansClusters(k=7)) # uses sklearn clustering
    #     # with minibatch, but no choice of distance measures
    # ])
    #
    # clusters = model.fit_transform(docs)
    #
    # # for idx, cluster in enumerate(clusters):
    # #     print("Document '{}' assigned to cluster {}.".format(pickles[idx],
    # #                                                          cluster))
    #
    # for idx, fileid in enumerate(pickles):
    #     print("Document '{}' assigned to cluster {}.".format(fileid,
    #                                                          clusters[idx]))

    # # Agglomerative hierarchical clustering pipeline
    # model = Pipeline([
    #     ("norm", Corpus_Vectorizer.TitleNormalizer()),
    #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
    #     ('clusters', HierarchicalClustering())
    # ])
    #
    # clusters = model.fit_transform(docs)
    # labels = model.named_steps['clusters'].labels
    #
    # for idx, fileid in enumerate(pickles):
    #     print("Document '{}' assigned to cluster {}.".format(fileid,
    #                                                          labels[idx]))

    # # Latent Dirchlicht Allocation
    # skmodel = SklearnTopicModels(n_components=3, estimator='LDA')
    #
    # skmodel.fit_transform(docs)
    # topics = skmodel.get_topics(n=5)
    # for topic, terms in topics.items():
    #     print("Topic #{}:".format(topic + 1))
    #     print(terms)

    # # Latent Semantic Allocation
    # skmodel = SklearnTopicModels(n_components=5, estimator='LSA')
    #
    # skmodel.fit_transform(docs)
    # topics = skmodel.get_topics(n=6)
    # for topic, terms in topics.items():
    #     print("Topic #{}:".format(topic + 1))
    #     print(terms)

    # Non-Negative Matrix Factorization
    skmodel = SklearnTopicModels(n_components=5, estimator='NMF')

    skmodel.fit_transform(docs)
    topics = skmodel.get_topics(n=6)
    for topic, terms in topics.items():
        print("Topic #{}:".format(topic + 1))
        print(terms)

