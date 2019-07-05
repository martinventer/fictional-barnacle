from unittest import TestCase

from CorpusProcessingTools import Context_Extraction
from CorpusReader import Elsevier_Corpus_Reader
from CorpusProcessingTools import Corpus_Vectorizer


class TestKeyphraseExtractor(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))


    def test_KMeansClusters(self):
        docs = list(self.corpus.title_tagged(fileids=self.subset))
        phrase_extractor = Context_Extraction.KeyphraseExtractor()
        keyphrases = list(phrase_extractor.fit_transform(docs))
        result = keyphrases[0]
        target = ['histologic evaluation of implants', 'flapless', 'surgery',
                  'study in canines']
        self.assertEqual(target, result)