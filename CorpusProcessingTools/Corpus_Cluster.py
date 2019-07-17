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

import numpy as np

from scipy import sparse

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
        if type(documents) is sparse.csr.csr_matrix:
            documents = documents.toarray()
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
            # ('vect', Corpus_Vectorizer.Text2FrequencyVector()),
            # ('vect', Corpus_Vectorizer.Text2OneHotVector()),
            ('vect', Corpus_Vectorizer.Text2TFIDVector()),
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
    from CorpusReaders import Elsevier_Corpus_Reader
    from CorpusProcessingTools import Corpus_Vectorizer
    from PlottingTools import Cluster_Plotting
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)

    # --------------------------------------------------------------------------
    # KMeansClusters
    # --------------------------------------------------------------------------
    raw = list(corpus.title_tagged())
    #
    # normalize = Corpus_Vectorizer.TextNormalizer()
    # norm = normalize.fit_transform(corpus.title_tagged())
    #
    # vectorize = Corpus_Vectorizer.Text2FrequencyVector()
    # vector = vectorize.fit_transform(norm)
    #
    # cluster = KMeansClusters()
    # clusters = cluster.fit_transform(vector)

    clusterer = Pipeline([('normalize', Corpus_Vectorizer.TextNormalizer()),
                         ('vectorize',
                         Corpus_Vectorizer.Text2FrequencyVector()),
                         ('cluster', KMeansClusters())])

    labels = clusterer.fit_transform(corpus.title_tagged())

    reduce = Pipeline([('normalize', Corpus_Vectorizer.TextNormalizer()),
                         ('vectorize',
                          Corpus_Vectorizer.Text2FrequencyVector()),
                         ('pca', TruncatedSVD(n_components=2))])

    X2d = reduce.fit_transform(corpus.title_tagged())

    Cluster_Plotting.plot_clusters(X2d, labels)
