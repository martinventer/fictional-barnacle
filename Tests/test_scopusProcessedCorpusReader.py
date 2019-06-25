from unittest import TestCase

from CorpusReader import Elsevier_Corpus_Reader


class TestScopusProcessedCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corp = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")

    def test_title_sents(self):
        self.assertEqual(next(self.corp.title_sents())[0],
                         ('Robots', 'NNS'))

    def test_title_tagged(self):
        self.assertEqual(next(self.corp.title_tagged()),
                         ('Robots', 'NNS'))

    def test_title_words(self):
        self.assertEqual(next(self.corp.title_words()),
                         'Robots')


