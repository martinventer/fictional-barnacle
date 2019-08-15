from unittest import TestCase

from CorpusReaders import Elsevier_Corpus_Reader, Corpus_Pre_Processor


class TestSplitCorpus(TestCase):
    def setUp(self) -> None:
        self.root = "Test_Corpus/Raw_corpus/"
        self.target = "Test_Corpus/Split_corpus/"
        self.corpus = Elsevier_Corpus_Reader.RawCorpusReader(root=self.root)

    def test_split_corpus(self):
        Corpus_Pre_Processor.split_corpus(corpus=self.corpus,
                                          target=self.target)
        new_corp = Elsevier_Corpus_Reader.RawCorpusReader(root=self.target)
        target = 14524
        result = len(list(new_corp.fileids()))
        self.assertEqual(target, result)


class TestScopusCorpusProcessor(TestCase):
    def setUp(self) -> None:
        self.root = "Test_Corpus/Split_corpus/"
        self.target = "Test_Corpus/Processed_corpus/"
        self.corpus = Elsevier_Corpus_Reader.ScopusCorpusReader(root=self.root)

    def test_split_corpus(self):
        processor = Corpus_Pre_Processor.ScopusCorpusProcessor(
            corpus=self.corpus,
            target=self.target)
        processor.transform()
        new_corp = Elsevier_Corpus_Reader.ScopusCorpusReader(root=self.target)
        result = next(new_corp.docs())
        target1 = ''
        self.assertEqual(result['processed:dc:title'], target1)
        target2 = ''
        self.assertEqual(result['processed:dc:description'], target2)
        target3 = ''
        self.assertEqual(result['file_name'], target3)
