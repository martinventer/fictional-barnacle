from unittest import TestCase

from CorpusProcessingTools import Corpus_Vectorizer


class TestTokenize(TestCase):
    def setUp(self) -> None:
        self.corpus = [
                "The elephant sneezed at the sight of potatoes.",
                "Bats can see via echolocation. SDee the bat sneeze!",
                "Wondering, she opened the door to the studio."
            ]

    def test_tokenize(self):
        target = ['the', 'eleph', 'sneez', 'at', 'the', 'sight', 'of', 'potato']
        result = [word for word in Corpus_Vectorizer.tokenize(self.corpus[0])]
        self.assertEqual(result, target)
        self.assertEqual(len(result), len(target))

