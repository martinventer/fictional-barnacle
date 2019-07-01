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


class KMeansClusters(BaseEstimator, TransformerMixin):
    """
    Cluster text data using k-means
    """
    def __init__(self, k=7):
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(self.k, self.distance,
                                     avoid_empty_clusters=True)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        return self.model.cluster(documents, assign_clusters=True)


class MiniBatchKMeansClusters(BaseEstimator, TransformerMixin):
    """
    Cluster text data using k-means
    """
    def __init__(self, k=7):
        self.k = k
        self.model = MiniBatchKMeans(self.k)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        return self.model.fit_predict(documents)




if __name__ == '__main__':
    from CorpusReader import Elsevier_Corpus_Reader
    from CorpusProcessingTools import Corpus_Vectorizer

    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    loader = Elsevier_Corpus_Reader.CorpusLoader(corpus, 12, shuffle=False)

    # docs = corpus.title_sents(categories='soft robot/2011')
    # docs = corpus.title_sents()

    # docs = list(corpus.title_sents(categories='soft robot/2011'))
    # pickles = list(corpus.fileids(categories='soft robot/2011'))
    #
    # print(len(docs), len(pickles))


    #
    # model = Pipeline([
    #     ("norm", Corpus_Vectorizer.TitleNormalizer2()),
    #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
    #     ('clusters', MiniBatchKMeansClusters(k=7))
    # ])
    #
    # clusters = model.fit_transform(docs)
    # # pickles = list(corpus.fileids())
    # pickles = list(corpus.fileids(categories='soft robot/2011'))
    #
    # # for idx, cluster in enumerate(clusters):
    # #     print("Document '{}' assigned to cluster {}.".format(pickles[idx],
    # #                                                          cluster))
    #
    #
    # # norm = Corpus_Vectorizer.TitleNormalizer2()
    # # norm.fit(docs)
    # # docs2 = norm.transform(docs)