from unittest import TestCase

import Transformers.Transformers
import Utils.Utils
from CorpusProcessingTools import Corpus_Vectorizer
from CorpusReaders import Elsevier_Corpus_Reader

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class TestTextStemTokenize(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")

    def test_stem(self):
        stemmed = Transformers.Transformers.TextStemTokenize()
        input_data = [
            [[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
              ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
              ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
              ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
              ('sensors', 'NNS')]]]
        targets = [['a', 'studi', 'of', 'laundri', 'tidi', 'laundri', 'state',
                    'determin', 'use', 'video', 'and', '3d', 'sensor']]
        for target, text in zip(targets, input_data):
            result = stemmed.stem(text)
            self.assertEqual(target, result)
            self.assertEqual(list, type(result))

    def test_transform(self):
        stemmed = Transformers.Transformers.TextStemTokenize()
        # check the that text is normalized
        input_data = [
            [[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
              ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
              ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
              ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
              ('sensors', 'NNS')]]]
        targets = [['a', 'studi', 'of', 'laundri', 'tidi', 'laundri', 'state',
                    'determin', 'use', 'video', 'and', '3d', 'sensor']]
        result = stemmed.transform(input_data)
        self.assertEqual(targets, result)

        # test that the output from the transformer works for all test titles
        result2 = stemmed.transform(self.corpus.title_tagged())
        self.assertEqual(list, type(result2))
        self.assertEqual(list, type(result2[0]))
        self.assertEqual(str, type(result2[0][0]))

        # test that the output from the transformer works for all test
        # descritions
        result3 = stemmed.transform(self.corpus.description_tagged())
        self.assertEqual(list, type(result3))
        self.assertEqual(list, type(result3[0]))
        self.assertEqual(str, type(result3[0][0]))


class TestTextNormalizer(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")

    def test_is_punct(self):
        normal = Corpus_Vectorizer
        input_data = ['.', ',', 'i', '?', 't.', '.t']
        targets = [True, True, False, True, False, False]
        for target, text in zip(targets, input_data):
            result = Utils.Utils.is_punct(text)
            self.assertEqual(target, result)
            self.assertEqual(bool, type(result))

    def test_is_stopword(self):
        normal = Transformers.Transformers.TextNormalizer()
        input_data = ['.', ',', 'i', '?', 't.', '.t', 'the', 'Steven']
        targets = [False, False, True, False, False, False, True, False]
        for target, text in zip(targets, input_data):
            result = normal.is_stopword(text)
            self.assertEqual(target, result)
            self.assertEqual(bool, type(result))

    def test_lemmatize(self):
        normal = Transformers.Transformers.TextNormalizer()
        targets = ['garden']
        input_data = [('gardening', 'V')]
        for target, text in zip(targets, input_data):
            result = normal.lemmatize(text[0], text[1])
            self.assertEqual(target, result)
            self.assertEqual(str, type(result))

    def test_normalize(self):
        normal = Transformers.Transformers.TextNormalizer()
        targets = [['study', 'laundry', 'tidiness', 'laundry', 'state',
                    'determination', 'use', 'video', '3d', 'sensor']]
        input_data = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'),
                   ('laundry', 'JJ'), ('tidiness', 'NN'), (':', ':'),
                   ('Laundry', 'JJ'), ('state', 'NN'), ('determination', 'NN'),
                   ('using', 'VBG'), ('video', 'NN'), ('and', 'CC'),
                   ('3D', 'CD'), ('sensors', 'NNS')]]]
        for target, text in zip(targets, input_data):
            result = normal.normalize(text)
            self.assertEqual(target, result)
            self.assertEqual(list, type(result))
            self.assertEqual(str, type(result[0]))

    def test_transform(self):
        normal = Transformers.Transformers.TextNormalizer()
        # check the that text is normalized
        targets = [['study', 'laundry', 'tidiness', 'laundry', 'state',
                    'determination', 'use', 'video', '3d', 'sensor']]
        input_data = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'),
                   ('laundry', 'JJ'), ('tidiness', 'NN'), (':', ':'),
                   ('Laundry', 'JJ'), ('state', 'NN'), ('determination', 'NN'),
                   ('using', 'VBG'), ('video', 'NN'), ('and', 'CC'),
                   ('3D', 'CD'), ('sensors', 'NNS')]]]
        result = normal.transform(input_data)
        self.assertEqual(targets, result)

        # test that the output from the transformer works for all test titles
        result2 = normal.transform(self.corpus.title_tagged())
        self.assertEqual(list, type(result2))
        self.assertEqual(list, type(result2[0]))
        self.assertEqual(str, type(result2[0][0]))

        # test that the output from the transformer works for all test
        # descritions
        result3 = normal.transform(self.corpus.description_tagged())
        self.assertEqual(list, type(result3))
        self.assertEqual(list, type(result3[0]))
        self.assertEqual(str, type(result3[0][0]))


class TestTextSimpleTokenizer(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")

    def test_get_words(self):
        simple = Transformers.Transformers.TextSimpleTokenizer()
        input_data = [
            [[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
              ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
              ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
              ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
              ('sensors', 'NNS')]]]
        targets = [['A', 'study', 'of', 'laundry', 'tidiness', ':', 'Laundry',
                    'state', 'determination', 'using', 'video', 'and', '3D',
                    'sensors']]
        for target, text in zip(targets, input_data):
            result = simple.get_words(text)
            self.assertEqual(target, result)
            self.assertEqual(list, type(result))

    def test_transform(self):
        simple = Transformers.Transformers.TextSimpleTokenizer()
        # check the that text is normalized
        input_data = [
            [[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
              ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
              ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
              ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
              ('sensors', 'NNS')]]]
        targets = [['A', 'study', 'of', 'laundry', 'tidiness', ':', 'Laundry',
                    'state', 'determination', 'using', 'video', 'and', '3D',
                    'sensors']]
        result = simple.transform(input_data)
        self.assertEqual(targets, result)

        # test that the output from the transformer works for all test titles
        result2 = simple.transform(self.corpus.title_tagged())
        self.assertEqual(list, type(result2))
        self.assertEqual(list, type(result2[0]))
        self.assertEqual(str, type(result2[0][0]))

        # test that the output from the transformer works for all test
        # descritions
        result3 = simple.transform(self.corpus.description_tagged())
        self.assertEqual(list, type(result3))
        self.assertEqual(list, type(result3[0]))
        self.assertEqual(str, type(result3[0][0]))


class TestCorpus2FrequencyVector(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")
        self.simple = Transformers.Transformers.TextSimpleTokenizer()
        self.input_text = self.simple.transform(self.corpus.title_tagged())

    def test_transform(self):
        vectorizer = Transformers.Transformers.Text2FrequencyVector()
        matrix = vectorizer.fit_transform(self.input_text)
        self.assertEqual(sparse.csr.csr_matrix, type(matrix))
        results = []
        for document in matrix:
            results.append(document.sum())

        targets = []
        for doc in self.corpus.document_title():
            tokens = [nltk.wordpunct_tokenize(sent)
                      for sent in nltk.sent_tokenize(doc)]
            tokens_flat = [item for sublist in tokens for item in sublist]
            targets.append(len(tokens_flat))

        self.assertEqual(len(targets), len(results))
        for result, target in zip(results, targets):
            self.assertEqual(target, result)


class TestText2OneHotVector(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")
        self.simple = Transformers.Transformers.TextSimpleTokenizer()
        self.input_text = self.simple.transform(self.corpus.title_tagged())

    def test_transform(self):
        vectorizer = Transformers.Transformers.Text2OneHotVector()
        matrix = vectorizer.fit_transform(self.input_text)
        self.assertEqual(sparse.csr.csr_matrix, type(matrix))
        results = []
        for document in matrix:
            results.append(document.sum())

        targets = []
        for doc in self.corpus.document_title():
            tokens = [nltk.wordpunct_tokenize(sent)
                      for sent in nltk.sent_tokenize(doc)]
            tokens_flat = set([item for sublist in tokens for item in sublist])
            targets.append(len(tokens_flat))

        self.assertEqual(len(targets), len(results))
        for result, target in zip(results, targets):
            self.assertEqual(target, result)


class TestText2TFIDVector(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")
        self.simple = Transformers.Transformers.TextSimpleTokenizer()
        self.tfidf = TfidfVectorizer(tokenizer=self._identity,
                                     preprocessor=None,
                                     lowercase=False)
        self.input_text = self.simple.transform(self.corpus.title_tagged())

    @staticmethod
    def _identity(words):
        return words

    def test_transform(self):
        vectorizer = Transformers.Transformers.Text2TFIDVector()
        matrix = vectorizer.fit_transform(self.input_text)
        self.assertEqual(sparse.csr.csr_matrix, type(matrix))
        results = []
        for document in matrix:
            results.append(document.sum())

        matrix = self.tfidf.fit_transform(self.input_text)
        targets = []
        for document in matrix:
            targets.append(document.sum())

        self.assertEqual(len(targets), len(results))
        for result, target in zip(results, targets):
            self.assertEqual(target, result)


class TestText2Doc2VecVector(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Test_Corpus/Processed_corpus/")
        self.simple = Transformers.Transformers.TextSimpleTokenizer()
        self.input_text = self.simple.transform(self.corpus.title_tagged())

    def test_transform(self):
        vectorizer = Transformers.Transformers.Text2Doc2VecVector()
        matrix = vectorizer.fit_transform(self.input_text)
        self.assertEqual(sparse.csr.csr_matrix, type(matrix))
