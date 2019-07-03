from unittest import TestCase

from CorpusProcessingTools import Corpus_Cluster
from CorpusReader import Elsevier_Corpus_Reader
from CorpusProcessingTools import Corpus_Vectorizer

from sklearn.pipeline import Pipeline

import warnings


class TestKMeansClusters(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))
        self.model = Pipeline([
            ("norm", Corpus_Vectorizer.TitleNormalizer()),
            ("vect", Corpus_Vectorizer.OneHotVectorizer()),
            ('clusters', Corpus_Cluster.KMeansClusters(k=7))
        ])

    def test_KMeansClusters(self):
        target = 4696
        docs = list(self.corpus.title_tagged(fileids=self.subset))
        result = self.model.fit_transform(docs)

        self.assertEqual(target, len(result))


class TestMiniBatchKMeansClusters(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))
        self.model = Pipeline([
            ("norm", Corpus_Vectorizer.TitleNormalizer()),
            ("vect", Corpus_Vectorizer.OneHotVectorizer()),
            ('clusters', Corpus_Cluster.MiniBatchKMeansClusters(k=7))
        ])

    def test_MiniBatchKMeansClusters(self):
        target = 4696
        docs = list(self.corpus.title_tagged(fileids=self.subset))
        result = self.model.fit_transform(docs)

        self.assertEqual(target, len(result))


class TestHierarchicalClustering(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))
        self.model = Pipeline([
            ("norm", Corpus_Vectorizer.TitleNormalizer()),
            ("vect", Corpus_Vectorizer.OneHotVectorizer()),
            ('clusters', Corpus_Cluster.HierarchicalClustering())
        ])

    def test_HierarchicalClustering(self):
        target = 4696
        docs = list(self.corpus.title_tagged(fileids=self.subset))
        result = self.model.fit_transform(docs)

        self.assertEqual(target, len(result))
