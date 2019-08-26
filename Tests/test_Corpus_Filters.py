from unittest import TestCase

from CorpusReaders import Elsevier_Corpus_Reader, Corpus_filters


class TestRawCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corp = Elsevier_Corpus_Reader.RawCorpusReader(
            "Test_Corpus/Raw_corpus/")
        self.all_files = self.corp.fileids()

    def test_affiliation_city(self):
        filtered_corpus = Corpus_filters.FilteredCorpus(self.corp)

        subset = filtered_corpus.affiliation_city()
        self.assertEqual(self.all_files, subset)
        # gen = self.corp.affiliation_city_s()
        # in_list = [next(gen) for i in range(10)]
        # out_list = [next(gen) for i in range(10)]
