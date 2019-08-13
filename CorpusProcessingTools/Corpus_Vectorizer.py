#! envs/fictional-barnacle/bin/python3.6
"""
Corpus_Vectorizer.py

@author: martinventer
@date: 2019-06-26

Tools for vectorizing text data
"""

import nltk
import string
import unicodedata

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from scipy import sparse


def identity(words):
    return words


def is_punct(token) -> bool:
    """
    given a token, determines whether it contains punctuation
    Parameters
    ----------
    token : str
        string token

    Returns
    -------
        Bool

    """
    return all(
        unicodedata.category(char).startswith('P') for char in token
    )


class TextStemTokenize(BaseEstimator, TransformerMixin):
    """
    Stems the tokens in a list of paragraphs list of sentances list of
    token, tag tuples, returning a list of stemmed words for each document
    """
    def __init__(self, language='english'):
        self.stemmer = nltk.stem.SnowballStemmer(language)

    def stem(self, doc) -> list:
        """
        takes a word makes it lower case, and returns the stemmed from.
        Parameters
        ----------
        doc : list
            list of paragraphs list of sentances list of token, tag tuples

        Returns
        -------
            a single list of stemmed words
        """
        word_list = []
        for sent in doc:
            for token, tag in sent:
                if token in string.punctuation:
                    continue
                word_list.append(self.stemmer.stem(token).lower())
        return word_list

    def fit(self, documents):
        return self

    def transform(self, documents):
        return [
            self.stem(document)
            for document in documents
        ]


class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    transformer that normalizes and lemmitizes text. THis transformer needs
    text in the form that one document is a list of sentences, which is a list
    of (token, tag) tuples.
    """
    def __init__(self, language='english'):
        """
        Initialize the text normalizer with the wordnet lemmatizer and the nltk
        stopwords in a selected language.
        Parameters
        ----------
        language : str
            string indication of the language
        """
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_stopword(self, token) -> bool:
        """
        returns a boolean True if a string is in the list of stopwords
        Parameters
        ----------
        token : str
            string token

        Returns
        -------
            Bool

        """
        return token.lower() in self.stopwords

    def lemmatize(self, token, pos_tag) -> str:
        """
        returns a lametized string
        Parameters
        ----------
        pos_tag : str
            string symobol containing the part of speach tag for a given word
        token : str
            string token

        Returns
        -------
            lematized word

        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def normalize(self, document) -> list:
        """
        returns a list of normalized words
        Parameters
        ----------
        document : list
            list of paragraphs that are lists of sentences that are lists of
            token tag tuples.

        Returns
        -------
            list of normalized words

        """
        return [
            self.lemmatize(token, tag).lower()
            for sentence in document
            for (token, tag) in sentence
            if not is_punct(token) and not self.is_stopword(token)
        ]

    def fit(self, documents):
        return self

    def transform(self, documents) -> list:
        """
        yields a list of normalized text for each document passed
        Parameters
        ----------
        documents: list of documents, or generator returning documents as as
        list of paragraps list of sents as (token, tag)

        Returns
        -------
            list of lametized lowercase text with punctuation removed
        """
        return [
            self.normalize(document)
            for document in documents
        ]


class CorpusSimpleNormalizer(BaseEstimator, TransformerMixin):
    """
    wrapper a  [[(token, tag]] corpus item that will return only lemmatized,
    lowercase words without punctuation
    """

    def __init__(self, language='english'):
        """
        Initialize the text normalizer with the wordnet lemmatizer and the nltk
        stopwords in a selected language.
        Parameters
        ----------
        language : str
            string indication of the language
        """
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_stopword(self, token) -> bool:
        """
        returns a boolean True if a string is in the list of stopwords
        Parameters
        ----------
        token : str
            string token

        Returns
        -------
            Bool

        """
        return token.lower() in self.stopwords

    def lemmatize(self, token, pos_tag) -> str:
        """
        returns a lametized string
        Parameters
        ----------
        pos_tag : str
            string symobol containing the part of speach tag for a given word
        token : str
            string token

        Returns
        -------
            lematized word

        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def normalize(self, document) -> list:
        """
        returns a list of normalized words
        Parameters
        ----------
        document : list
            list of paragraphs that are lists of sentences that are lists of
            token tag tuples.

        Returns
        -------
            list of normalized words

        """
        return [
            self.lemmatize(token, tag).lower()
            for sentence in document
            for (token, tag) in sentence
            if not is_punct(token) and not self.is_stopword(token)
        ]

    def fit(self, documents):
        return self

    def transform(self, documents):
        for document in documents:
            yield [term for term in self.normalize(document)]


class TextSimpleTokenizer(BaseEstimator, TransformerMixin):
    """
    transformer that takes text in the form of list of sentances that are
    lists of token, tag tuples and returns a single list of strings for each
    document
    """
    def __init__(self):
        """

        """
        pass

    def get_words(self, document):
        return [
            token
            for sentance in document
            for (token, tag) in sentance
        ]

    def fit(self, documents):
        return self

    def transform(self, documents):
        return [
            self.get_words(document)
            for document in documents
        ]


class Text2wordCountVector(CountVectorizer):
    """
    Wrapper for CountVectoriser that converts a corpus into a matrix of term
    occurrences per document. Requires the input data to be in the form of
    one list of words per document. does not make use of a preprocessor of
    force lowercase
    """
    def __init__(self):
        # CountVectorizer.__init__(self, tokenizer=identity, preprocessor=None)
        CountVectorizer.__init__(self, tokenizer=identity,
                                 preprocessor=None,
                                 lowercase=False)


class Text2FrequencyVector(CountVectorizer):
    """
    Wrapper for CountVectoriser that converts a corpus into a matrix of term
    occurrences per document. Requires the input data to be in the form of
    one list of words per document.
    """
    def __init__(self):
        CountVectorizer.__init__(self, tokenizer=identity,
                                 preprocessor=None,
                                 lowercase=False)


class Text2OneHotVector(CountVectorizer):
    """
    Wrapper for CountVectoriser that converts a corpus into a matrix of one
    hot vectors per document. Requires the input data to be in the form of
    one list of words per document. This can be done using
    """
    def __init__(self):
        CountVectorizer.__init__(self, binary=True,
                                 tokenizer=identity,
                                 preprocessor=None,
                                 lowercase=False)


class Text2TFIDVector(TfidfVectorizer):
    """
    Wrapper for TfidfVectorizer that converts a corpus into a matrix of
    tifidf vectors per document. Requires the input data to be in the form of
    one list of words per document. This can be done using    """
    def __init__(self):
        TfidfVectorizer.__init__(self, tokenizer=identity,
                                 preprocessor=None,
                                 lowercase=False)


class Text2Doc2VecVector(BaseEstimator, TransformerMixin):
    """
    transformer that converts a corpus into a matrix of
    doc2vec vectors per document. Requires the input data to be in the form of
    one list of words per document. This can be done using one of the text
    normalizers
    """
    def __init__(self, vector_size=50, min_count=0):
        """
        Parameters
        ----------
        vector_size : int
            Desired vector length for output
        min_count : int
            Doc2Vec will ignore any tokens with a count below this number

        """
        self.min_count = min_count
        self.vector_size = vector_size

    @staticmethod
    def gensim_docs(documents) -> list:
        """
        Convert the raw input observations to a tagged document list
        Parameters
        ----------
        documents : list

        Returns
        -------
            tagged document list
        """
        docs = [
            TaggedDocument(words, ['d{}'.format(idx)])
            for idx, words in enumerate(documents)
        ]
        return docs

    def fit(self, documents):
        docs = self.gensim_docs(documents)
        model = Doc2Vec(docs,
                        vector_size=self.vector_size,
                        min_count=self.min_count)
        self.model = model
        return self

    def transform(self, documents):
        temp = sparse.csr_matrix(self.model.docvecs.doctag_syn0)
        # return self.model.docvecs, temp
        return temp


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)

    # --------------------------------------------------------------------------
    # TextNormalizer
    # --------------------------------------------------------------------------
    normal = TextNormalizer()

    input = ['.', ',', 'i', '?', 't.', '.t']
    for word in input:
        print(word, is_punct(word))

    input = ['.', ',', 'i', '?', 't.', '.t', 'the', 'Steven']
    for word in input:
        print(word, normal.is_stopword(word))

    input = [('gardening', 'V')]
    for token, tag in input:
        print(token, tag, normal.lemmatize(token, tag))

    input = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
              ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
              ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
              ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
              ('sensors', 'NNS')]]]
    for doc in input:
        print(normal.normalize(doc))

    input = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
               ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
               ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
               ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
               ('sensors', 'NNS')]]]
    print(normal.fit_transform(input))

    aa = normal.transform(corpus.title_tagged())

    # --------------------------------------------------------------------------
    # TextStemTokenize
    # --------------------------------------------------------------------------
    # stemmed = TextStemTokenize()
    #
    # input = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
    #           ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
    #           ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
    #           ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
    #           ('sensors', 'NNS')]]]
    # for doc in input:
    #     print(stemmed.stem(doc))

    # --------------------------------------------------------------------------
    # TextSimpleTokenizer
    # --------------------------------------------------------------------------
    # simple = TextSimpleTokenizer()
    #
    # input = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
    #           ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
    #           ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
    #           ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
    #           ('sensors', 'NNS')]]]
    # for doc in input:
    #     print(simple.get_words(doc))

    # --------------------------------------------------------------------------
    # Text2FrequencyVector
    # --------------------------------------------------------------------------
    # simple = TextSimpleTokenizer()
    # vec = Text2FrequencyVector()
    #
    # input_text = [['From', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
    #               'realization','of', 'qualitative', 'artificial',
    #                'intelligence', 'based', 'on', 'task','realizability',
    #                'analysis']]
    # vector = vec.fit_transform(input_text)
    # for row in vector:
    #     print(row.sum())
    #
    # input_text = simple.transform(corpus.title_tagged())
    # vector = vec.fit_transform(input_text)
    #
    # print("{} documents, {} words".format(vector.shape[0], vector.shape[1]))
    # print(type(vector))

    # --------------------------------------------------------------------------
    # Text2OneHotVector
    # --------------------------------------------------------------------------
    # simple = TextSimpleTokenizer()
    # vec = Text2OneHotVector()
    #
    # input_text = [['From', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
    #                'realization', 'of', 'qualitative', 'artificial',
    #                'intelligence', 'based', 'on', 'task', 'realizability',
    #                'analysis', 'analysis']]
    # vector = vec.fit_transform(input_text)
    # for row in vector:
    #     print(row.sum())
    #
    # input_text = simple.transform(corpus.title_tagged())
    # vector = vec.fit_transform(input_text)
    #
    # print("{} documents, {} words".format(vector.shape[0], vector.shape[1]))
    # print(type(vector))

    # --------------------------------------------------------------------------
    # Text2TFIDVector
    # --------------------------------------------------------------------------
    # simple = TextSimpleTokenizer()
    # vec = Text2TFIDVector()
    #
    # input_text = [['From', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
    #                'realization', 'of', 'qualitative', 'artificial',
    #                'intelligence', 'based', 'on', 'task', 'realizability',
    #                'analysis', 'analysis']]
    # vector = vec.fit_transform(input_text)
    # for row in vector:
    #     print(row)
    #
    # input_text = simple.transform(corpus.title_tagged())
    # vector = vec.fit_transform(input_text)
    #
    # print("{} documents, {} words".format(vector.shape[0], vector.shape[1]))
    # print(type(vector))

    # --------------------------------------------------------------------------
    # Text2Doc2VecVector
    # --------------------------------------------------------------------------
    # simple = TextSimpleTokenizer()
    # vec = Text2Doc2VecVector(vector_size=3,
    #                           min_count=0)
    #
    # input_text = [['From', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
    #                'realization', 'of', 'qualitative', 'artificial',
    #                'intelligence', 'based', 'on', 'task', 'realizability',
    #                'analysis', 'analysis'],
    #               ['try', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
    #                'realization', 'of', 'qualitative', 'artificial',
    #                'intelligence', 'based', 'on', 'task', 'realizability',
    #                'analysis']
    #               ]
    # vector = vec.fit_transform(input_text)
    # for row in vector:
    #     print(row)
    #
    # input_text = simple.transform(corpus.title_tagged())
    # vector = vec.fit_transform(input_text)
    #
    # print("{} documents, {} components".format(vector.shape[0], vector.shape[
    #     1]))

