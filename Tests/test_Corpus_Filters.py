from unittest import TestCase

from CorpusReaders import Elsevier_Corpus_Reader
from CorpusFilterTools import Corpus_filters


class TestCorpus2Frame(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))

    def test_transform(self):
        c_framer = Corpus_filters.Corpus2Frame()
        df = c_framer.transform(self.corpus, fileids=self.subset)
        target = (4696, 36)
        result = df.shape
        self.assertEqual(target, result)
