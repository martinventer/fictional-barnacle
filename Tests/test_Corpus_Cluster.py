from unittest import TestCase

import TextTools.Transformers
from Depricated import Corpus_Cluster, Corpus_Vectorizer
from CorpusReaders import Elsevier_Corpus_Reader

from sklearn.pipeline import Pipeline


class TestKMeansClusters(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))
        self.model = Pipeline([
            ("norm", Corpus_Vectorizer.TitleNormalizer()),
            ("vect", Corpus_Vectorizer.OneHotVectorizer()),
            ('clusters', TextTools.Transformers.KMeansClusters(k=7))
        ])

    def test_KMeansClusters(self):
        target = 4696
        docs = list(self.corpus.title_tagged(fileids=self.subset))
        result = self.model.fit_transform(docs)

        self.assertEqual(target, len(result))


class TestMiniBatchKMeansClusters(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")
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
            "Test_Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))
        self.model = Pipeline([
            ("norm", Corpus_Vectorizer.TitleNormalizer()),
            ("vect", Corpus_Vectorizer.OneHotVectorizer()),
            ('clusters', TextTools.Transformers.HierarchicalClustering())
        ])

    def test_HierarchicalClustering(self):
        target = 4696
        docs = list(self.corpus.title_tagged(fileids=self.subset))
        result = self.model.fit_transform(docs)

        self.assertEqual(target, len(result))


class TestSklearnTopicModels(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))

    def test_transform(self):
        docs = list(self.corpus.title_tagged(fileids=self.subset))
        skmodel = TextTools.Transformers.SklearnTopicModels(n_components=3,
                                                            estimator='LDA')
        skmodel.fit_transform(docs)
        topics = skmodel.get_topics(n=5)
        target = 3
        result = topics[0]
        self.assertEqual(target, len(result))
        target = list
        self.assertEqual(target, type(result))