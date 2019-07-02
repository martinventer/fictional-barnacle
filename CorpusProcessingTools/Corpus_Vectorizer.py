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
from sklearn.feature_extraction.text import CountVectorizer, \
    HashingVectorizer, TfidfVectorizer

from sklearn.base import BaseEstimator, TransformerMixin


def tokenize(text) -> list:
    """
    takes a string input and converts it to a tokenized list of words
    Parameters
    ----------
    text : str
        the text to be converted

    Returns
    -------
        list
    """
    stem = nltk.stem.SnowballStemmer("english")
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation:
            continue
        yield stem.stem(token)


class CorpusFrequencyVector(CountVectorizer, HashingVectorizer):
    """
    Wrapper for CountVectoriser or HashingVectorizer
    """
    def __init__(self, large_file=False):
        if large_file:
            HashingVectorizer.__init__(self)
        else:
            CountVectorizer.__init__(self)


class CorpusOneHotVector(CountVectorizer):
    """
    wrapper for CountVectorizer that returns only one hot encoding
    """
    def __init__(self):
        CountVectorizer.__init__(self, binary=True)


class OneHotVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.vectorizer = CountVectorizer(binary=True)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        freqs = self.vectorizer.fit_transform(documents)
        return [freq.toarray()[0] for freq in freqs]


class CorpusTFIDVector(TfidfVectorizer):
    """
    wrapper for TfidVectorizer
    """
    def __init__(self):
        TfidfVectorizer.__init__(self)


class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    transformer that normalizes and lemmitizes text. THis transformer needs
    text in the form that one documnet is a list of parragraphs, which is a
    list of sentences, which is a list of (token, tag) tuples.
    """
    def __init__(self, language='english'):
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            # yield self.normalize(document[0])
            yield self.normalize(document)


class TitleNormalizer(TextNormalizer):
    """
    Varient of TextNormalizer that returns a single string containing only a
    normalize single string of the title.
    requires the titles in the form [title[sentences[(token, tagged)]]]
    """
    def __init__(self, **kwargs):
        TextNormalizer.__init__(self, **kwargs)

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            # for title in document
            for sentence in document
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def transform(self, documents):
        return [" ".join(self.normalize(doc)) for doc in documents]


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
    from CorpusReader import Elsevier_Corpus_Reader

    # corpus = [
    #     "The elephant sneezed at the sight of potatoes.",
    #     "Bats can see via echolocation. See the bat sneeze!",
    #     "Wondering, she opened the door to the studio."
    # ]
    #
    # test = [word for word in tokenize(corpus[0])]

    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")
    loader = Elsevier_Corpus_Reader.CorpusLoader(corpus, 12, shuffle=False)

    docs = list(corpus.title_tagged(fileids=loader.fileids(1, test=True)))
    labels = loader.labels(0, test=True)

    # normal = TextNormalizer()
    # normal.fit(docs, labels)
    # print(list(normal.transform(docs))[0])

    normal = TitleNormalizer()
    normal.fit(docs, labels)
    result = list(normal.transform(docs))



