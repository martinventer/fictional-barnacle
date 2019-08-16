from unittest import TestCase

import TextTools.Transformers
from Depricated import Context_Extraction
from CorpusReaders import Elsevier_Corpus_Reader


class TestKeyphraseExtractor(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")

    def test_transform(self):
        phrase_extractor = TextTools.Transformers.KeyphraseExtractorL()
        phrases = phrase_extractor.fit_transform(self.corpus.title_tagged())
        for doc in phrases:
            self.assertEqual(list, type(doc))


class TestEntityExtractor(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))

    def test_transform(self):
        docs = list(self.corpus.title_tagged(fileids=self.subset))
        phrase_extractor = TextTools.Transformers.EntityExtractor()
        keyphrases = list(phrase_extractor.fit_transform(docs))
        result = keyphrases[0]
        target = ['histologic']
        self.assertEqual(target, result)


class TestRankGrams(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))

    def test_transform_n2(self):
        docs = list(self.corpus.title_words(fileids=self.subset))
        ranker = TextTools.Transformers.RankGrams(n=2)
        ranked = list(ranker.transform(docs))
        result = ranked[0][0]
        target = ('based', 'on')
        self.assertEqual(target, result)
        result = len(ranked[0][0])
        target = 2
        self.assertEqual(target, result)

    def test_transform_n3(self):
        docs = list(self.corpus.title_words(fileids=self.subset))
        ranker = TextTools.Transformers.RankGrams(n=3)
        ranked = list(ranker.transform(docs))
        result = ranked[0][0]
        target = ('based', 'on', 'the')
        self.assertEqual(target, result)
        result = len(ranked[0][0])
        target = 3
        self.assertEqual(target, result)

    def test_transform_n4(self):
        docs = list(self.corpus.title_words(fileids=self.subset))
        ranker = TextTools.Transformers.RankGrams(n=4)
        ranked = list(ranker.transform(docs))
        result = ranked[0][0]
        target = ('based', 'on', 'multi', '-')
        self.assertEqual(target, result)
        result = len(ranked[0][0])
        target = 4
        self.assertEqual(target, result)