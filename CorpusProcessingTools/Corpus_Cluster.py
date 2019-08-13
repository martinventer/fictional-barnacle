#! envs/fictional-barnacle/bin/python3.6
"""
Corpus_Vectorizer.py

@author: martinventer
@date: 2019-06-28

Tools Clustering corpus
"""

import nltk
from nltk.cluster import KMeansClusterer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF

import numpy as np
from scipy import sparse

from CorpusProcessingTools import Corpus_Vectorizer


class KMeansClusters(BaseEstimator, TransformerMixin):
    """
    Cluster text data using k-means. Makes use of nltk k-means clustering.
    Allows for alternative distance measures
    """
    def __init__(self, k=7, distance=None):
        """
        initializes the kmeans clustering
        Parameters
        ----------
        k : int
            number of clusters desired
        distance :
            an nltk.cluster.util for an alternative distance measure
        """
        self.k = k
        if not distance:
            self.distance = nltk.cluster.util.cosine_distance
        else:
            self.distance = distance
        self.model = KMeansClusterer(self.k, self.distance,
                                     avoid_empty_clusters=True)

    def fit(self, text_vector):
        return self

    def transform(self, text_vector):
        """
        fits the K-means model to the given documents
        Parameters
        ----------
        text_vector :
            a matrix of vectorised documents, each row contains a vector for
            each document

        Returns
        -------
            ndarray of document labels
        """
        if type(text_vector) is sparse.csr.csr_matrix:
            text_vector = text_vector.toarray()
        return np.array(self.model.cluster(text_vector, assign_clusters=True))


class HierarchicalClustering(BaseEstimator, TransformerMixin):
    """
    wrapper for the
    """

    def __init__(self, **kwargs):
        self.model = AgglomerativeClustering(**kwargs)

    def fit(self, text_vector):
        return self

    def transform(self, text_vector):
        """
        fits an agglomerative clustering to given vector
        Parameters
        ----------
        text_vector :
            a matrix of vectorised documents, each row contains a vector for
            each document

        Returns
        -------
            ndarray of document labels

        """
        if type(text_vector) is sparse.csr.csr_matrix:
            text_vector = text_vector.toarray()

        return np.array(self.model.fit_predict(text_vector))


class SklearnTopicModels(BaseEstimator, TransformerMixin):
    """
    a topic modeler that identifies the main topics in a corpus of documents
    """

    def __init__(self, n_topics=50, estimator='LDA', vectorizor="tfidvec"):
        """
        a topic modler calling form sklearn decomposition libray
        Parameters
        ----------
        n_topics : int
            number of topics assigned to the
        estimator : str
            'LDA' Latent Dirichlet Allocation (default)
            'LSA' Latent Semantic Analysis
            'NMF' Non-Negative Matrix Factorization
        vectorizor : str
            'freqvec' term frequency vector
            'tfidvec' term frequency-inverse document frequency vector
            'onehotvec' term frequency vector with one hot encoding
        """
        self.n_topics = n_topics

        if estimator is not "LDA":
            if estimator == 'LSA':
                self.estimator = TruncatedSVD(n_components=self.n_topics)
            elif estimator == 'NMF':
                self.estimator = NMF(n_components=self.n_topics)
        else:
            self.estimator = LatentDirichletAllocation(
                n_components=self.n_topics)

        if vectorizor is not 'tfidvec':
            if vectorizor is 'freqvec':
                self.vectorizor = Corpus_Vectorizer.Text2FrequencyVector()
            elif vectorizor is 'onehotvec':
                self.vectorizor = Corpus_Vectorizer.Text2OneHotVector()

        else:
            self.vectorizor = Corpus_Vectorizer.Text2TFIDVector()

        self.model = Pipeline([
            ('norm', Corpus_Vectorizer.TextNormalizer()),
            ('vect', self.vectorizor),
            ('model', self.estimator)
        ])

    def fit(self, documents):
        return self

    def transform(self, documents):
        self.model.fit_transform(documents)

        return self.model

    def get_topics(self, n=25):
        """
        n_terms is the number of top terms to show for each topic
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
    from PlottingTools import Cluster_Plotting
    from time import time

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)
    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 10, shuffle=False)
    subset_fileids = next(loader.fileids(test=True))

    # --------------------------------------------------------------------------
    # KMeansClusters
    # --------------------------------------------------------------------------
    docs = list(corpus.title_tagged(fileids=subset_fileids))
    observations = list(corpus.title_tagged(fileids=subset_fileids))

    # Text2FrequencyVector
    prepare_data = Pipeline([('normalize', Corpus_Vectorizer.TextNormalizer()),
                             ('vectorize',
                              Corpus_Vectorizer.Text2FrequencyVector())])
    X = prepare_data.fit_transform(observations)

    model = KMeansClusters(k=10)
    labels = model.fit_transform(X)

    Cluster_Plotting.plot_clusters_2d(X, labels)

    # Text2OneHotVector
    # prepare_data = Pipeline([('normalize', Corpus_Vectorizer.TextNormalizer()),
    #                          ('vectorize',
    #                           Corpus_Vectorizer.Text2OneHotVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # model = KMeansClusters(k=10)
    # labels = model.fit_transform(X)
    #
    # Cluster_Plotting.plot_clusters_2d(X, labels)

    # Text2Doc2VecVector
    # prepare_data = Pipeline([('normalize', Corpus_Vectorizer.TextNormalizer()),
    #                          ('vectorize',
    #                           Corpus_Vectorizer.Text2Doc2VecVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # model = KMeansClusters(k=10)
    # labels = model.fit_transform(X)
    #
    # Cluster_Plotting.plot_clusters_2d(X, labels)

    # Text2TFIDVector
    # prepare_data = Pipeline([('normalize', Corpus_Vectorizer.TextNormalizer()),
    #                          ('vectorize',
    #                           Corpus_Vectorizer.Text2TFIDVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # model = KMeansClusters(k=10)
    # labels = model.fit_transform(X)
    #
    # Cluster_Plotting.plot_clusters_2d(X, labels)

    # --------------------------------------------------------------------------
    # HierarchicalClustering
    # --------------------------------------------------------------------------
    # observations = list(corpus.title_tagged(fileids=subset_fileids))

    # Text2FrequencyVector
    # prepare_data = Pipeline([('normalize', Corpus_Vectorizer.TextNormalizer()),
    #                          ('vectorize',
    #                           Corpus_Vectorizer.Text2FrequencyVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # for linkage in ('ward', 'average', 'complete', 'single'):
    #     clustering = HierarchicalClustering(linkage=linkage, n_clusters=10)
    #     t0 = time()
    #     labels = clustering.fit_transform(X)
    #     print("%s :\t%.2fs" % (linkage, time() - t0))
    #
    #     Cluster_Plotting.plot_clusters_2d(X, labels)

    # Text2OneHotVector
    # prepare_data = Pipeline(
    #     [('normalize', Corpus_Vectorizer.TextNormalizer()),
    #      ('vectorize',
    #       Corpus_Vectorizer.Text2OneHotVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # for linkage in ('ward', 'average', 'complete', 'single'):
    #     clustering = HierarchicalClustering(linkage=linkage, n_clusters=10)
    #     t0 = time()
    #     labels = clustering.fit_transform(X)
    #     print("%s :\t%.2fs" % (linkage, time() - t0))
    #
    #     Cluster_Plotting.plot_clusters_2d(X, labels)

    # Text2TFIDVector
    # prepare_data = Pipeline(
    #     [('normalize', Corpus_Vectorizer.TextNormalizer()),
    #      ('vectorize',
    #       Corpus_Vectorizer.Text2TFIDVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # for linkage in ('ward', 'average', 'complete', 'single'):
    #     clustering = HierarchicalClustering(linkage=linkage, n_clusters=10)
    #     t0 = time()
    #     labels = clustering.fit_transform(X)
    #     print("%s :\t%.2fs" % (linkage, time() - t0))
    #
    #     Cluster_Plotting.plot_clusters_2d(X, labels)

    # Text2Doc2VecVector
    prepare_data = Pipeline(
        [('normalize', Corpus_Vectorizer.TextNormalizer()),
         ('vectorize',
          Corpus_Vectorizer.Text2Doc2VecVector())])
    X = prepare_data.fit_transform(docs)

    for linkage in ('ward', 'average', 'complete', 'single'):
        clustering = HierarchicalClustering(linkage=linkage, n_clusters=10)
        t0 = time()
        labels = clustering.fit_transform(X)
        print("%s :\t%.2fs" % (linkage, time() - t0))

        Cluster_Plotting.plot_clusters_2d(X, labels)

    # --------------------------------------------------------------------------
    # SklearnTopicModels
    # --------------------------------------------------------------------------
    # observations = list(corpus.title_tagged(fileids=subset_fileids))
    #
    # """
    # options
    # LDA , LSA,  NMF
    # 'freqvec','tfidvec','onehotvec'
    # """
    #
    # model = SklearnTopicModels(n_topics=5, estimator='NMF',
    #                            vectorizor="tfidvec")
    #
    # model.fit_transform(observations)
    # topics = model.get_topics(n_terms=10)
    # for topic, terms in topics.items():
    #     print("Topic #{} \t:".format(topic))
    #     print(terms)
