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


# def tokenize(text) -> list:
#     """
#     takes a string input and converts it to a tokenized list of words
#     Parameters
#     ----------
#     text : str
#         the text to be converted
#
#     Returns
#     -------
#         list
#     """
#     stem = nltk.stem.SnowballStemmer("english")
#     text = text.lower()
#
#     for token in nltk.word_tokenize(text):
#         if token in string.punctuation:
#             continue
#         yield stem.stem(token)


class TextStemTokenize(BaseEstimator, TransformerMixin):
    """
    Stems the tokens in a list of paragraphs list of sentances list of
    token, tag tuples
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

    def is_punct(self, token) -> bool:
        """
        returns a boolean True if a string is punctuation
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
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def fit(self, documents):
        return self

    def transform(self, documents):
        return [
            self.normalize(document)
            for document in documents
        ]


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


# class TitleNormalizer(TextNormalizer):
#     """
#     Varient of TextNormalizer that returns a single string containing only a
#     normalize single string of the title.
#     requires the titles in the form [title[sentences[(token, tagged)]]]
#     """
#     def __init__(self, **kwargs):
#         TextNormalizer.__init__(self, **kwargs)
#
#     def normalize(self, document) -> list:
#         return [
#             self.lemmatize(token, tag).lower()
#             # for title in document
#             for sentence in document
#             for (token, tag) in sentence
#             if not self.is_punct(token) and not self.is_stopword(token)
#         ]
#
#     def transform(self, documents):
#         return [" ".join(self.normalize(doc)) for doc in documents]


class Text2FrequencyVector(CountVectorizer):
    """
    Wrapper for CountVectoriser that converts a corpus into a matrix of term
    occurrences per document. Requires the input data to be in the form of
    one list of words per document. This can be done using
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
    tifidf vectors per document. Requires the input data to be in the form of
    one list of words per document. This can be done using one of the text
    normalizers
    """
    def __init__(self, vector_size=5, min_count=0):
        """
        Parameters
        ----------
        size : int
            Desired vector length for output
        min_count : int
            Doc2Vec will ignore any tokens with a count below this number

        """
        self.min_count = min_count
        self.vector_size = vector_size

    def gensim_docs(self, documents) -> list:
        """
        Convert the raw input docs to a tagged document list
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
        return self.model.docvecs

# class OneHotVectorizer(BaseEstimator, TransformerMixin):
#
#     def __init__(self):
#         self.vectorizer = CountVectorizer(binary=True)
#
#     def fit(self, documents, labels=None):
#         return self
#
#     def transform(self, documents):
#         freqs = self.vectorizer.fit_transform(documents)
#         return [freq.toarray()[0] for freq in freqs]


# class TitleNormalizer2(TitleNormalizer):
#     """
#     adapted TitleNormalizer that returns a string insted of a list
#     """
#     def __init__(self, **kwargs):
#         TextNormalizer.__init__(self, **kwargs)
#
#     def normalize(self, document):
#         return [
#             self.lemmatize(token, tag).lower()
#             for (token, tag) in document
#             if not self.is_punct(token) and not self.is_stopword(token)
#         ]
#
#     def transform(self, documents):
#         return [" ".join(self.normalize(doc)) for doc in documents]


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)

    # --------------------------------------------------------------------------
    # TextNormalizer
    # --------------------------------------------------------------------------
    # normal = TextNormalizer()

    # input = ['.', ',', 'i', '?', 't.', '.t']
    # for word in input:
    #     print(word, normal.is_punct(word))

    # input = ['.', ',', 'i', '?', 't.', '.t', 'the', 'Steven']
    # for word in input:
    #     print(word, normal.is_stopword(word))

    # input = [('gardening', 'V')]
    # for token, tag in input:
    #     print(token, tag, normal.lemmatize(token, tag))

    # input = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
    #           ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
    #           ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
    #           ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
    #           ('sensors', 'NNS')]]]
    # for doc in input:
    #     print(normal.normalize(doc))

    # input = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
    #            ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
    #            ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
    #            ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
    #            ('sensors', 'NNS')]]]
    # print(normal.transform(input))
    #
    # aa = normal.transform(corpus.title_tagged())

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
    # print(type(vector))

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
    simple = TextSimpleTokenizer()
    vec = Text2Doc2VecVector(vector_size=3,
                              min_count=0)

    input_text = [['From', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
                   'realization', 'of', 'qualitative', 'artificial',
                   'intelligence', 'based', 'on', 'task', 'realizability',
                   'analysis', 'analysis'],
                  ['try', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
                   'realization', 'of', 'qualitative', 'artificial',
                   'intelligence', 'based', 'on', 'task', 'realizability',
                   'analysis']
                  ]
    vector = vec.fit_transform(input_text)
    print(vector[0])
    print(vector[1])

    input_text = simple.transform(corpus.title_tagged())
    vector = vec.fit_transform(input_text)

    print("{} documents, {} words".format(vector.shape[0], vector.shape[1]))
    print(type(vector))

    print(vector[0])
    print(vector[1])

    sparse.csr_matrix(vector)
    import numpy as np
    test = vector.astype()
