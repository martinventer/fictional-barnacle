from unittest import TestCase

from CorpusProcessingTools import Corpus_Vectorizer
from CorpusReader import Elsevier_Corpus_Reader


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


class TestTitleNormalizer(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))

    def test_titleNormalizer(self):
        target = 'histologic evaluation implant follow flapless flap surgery ' \
                 'study canine'
        docs = list(self.corpus.title_tagged(fileids=self.subset))
        labels = [
            self.corpus.categories(fileids=fileid)[0]
            for fileid in self.subset
        ]
        normal = Corpus_Vectorizer.TitleNormalizer()
        normal.fit(docs, labels)
        result = list(normal.transform(docs))[0]

        self.assertEqual(result, target)


class TestTextNormalizer(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corpus,
                                                              n_folds=12,
                                                              shuffle=False)
        self.subset = next(self.loader.fileids(test=True))

    def test_transform(self):
        target = ['histologic', 'evaluation', 'implant', 'follow', 'flapless',
                  'flap', 'surgery', 'study', 'canine']
        docs = list(self.corpus.title_tagged(fileids=self.subset))
        labels = [
            self.corpus.categories(fileids=fileid)[0]
            for fileid in self.subset
        ]
        normal = Corpus_Vectorizer.TextNormalizer()
        normal.fit(docs, labels)
        result = list(normal.transform(docs))[0]

        self.assertEqual(result, target)
