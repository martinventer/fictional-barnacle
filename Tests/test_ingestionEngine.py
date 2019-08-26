from unittest import TestCase

from CorpusReaders import Elsivier_Ingestor


class TestScopusIngestionEngine(TestCase):
    def setUp(self) -> None:
        self.ingestor = Elsivier_Ingestor.ScopusIngestionEngine(
            file_path="Test_Corpus_download/Raw_corpus/",
            home=False,
            batch_size=25)

    def test_build_corpus_no_terms(self):
        result = self.ingestor.build_corpus(search_terms=[],
                                            dates=(1998, 1999))
        target = 'FAIL'
        self.assertEqual(target, result)

    def test_build_corpus_no_date(self):
        result = self.ingestor.build_corpus(search_terms=['shoes', 'socks'],
                                            dates=(1998,))
        target = 'FAIL'
        self.assertEqual(target, result)

    def test_build_corpus(self):
        result = self.ingestor.build_corpus(search_terms=['socks'],
                                            dates=(1998, 1999))
        target = 'PASS'
        self.assertEqual(target, result)


class TestSciDirIngestionEngine(TestCase):
    def setUp(self) -> None:
        self.ingestor = Elsivier_Ingestor.SciDirIngestionEngine(
            file_path="Test_Corpus_download/Raw_corpus/",
            home=False,
            batch_size=25)

    def test_build_corpus_no_terms(self):
        result = self.ingestor.build_corpus(search_terms=[],
                                            dates=(1998, 1999))
        target = 'FAIL'
        self.assertEqual(target, result)

    def test_build_corpus_no_date(self):
        result = self.ingestor.build_corpus(search_terms=['shoes', 'socks'],
                                            dates=(1998,))
        target = 'FAIL'
        self.assertEqual(target, result)

    def test_build_corpus(self):
        result = self.ingestor.build_corpus(search_terms=['socks'],
                                            dates=(1998, 1999))
        target = 'PASS'
        self.assertEqual(target, result)
