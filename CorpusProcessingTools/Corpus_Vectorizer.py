#! envs/fictional-barnacle/bin/python3.6
"""
Corpus_Vectorizer.py

@author: martinventer
@date: 2019-06-26

Tools for vectorizing text data
"""

import nltk
import string
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, \
    HashingVectorizer, TfidfVectorizer


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


def frequency_vector(text, large_file=False, binary=False, tfid=False) -> \
        bytearray:
    """
    a simple word frequency vectorisor for text
    Parameters
    ----------
    tfid
    binary : bool
        flag for one_hot encoding
    text :
        a list of text to be vectorised
    large_file : bool
        a flag to switch between the conventional CountVecotoriser and the
        Hash Vectorizer

    Returns
    -------
        sparse matrix

    """
    vectorizor = CountVectorizer(binary=binary)
    if large_file:
        vectorizor = HashingVectorizer()
    if tfid:
        vectorizor = TfidfVectorizer()
    return vectorizor.fit_transform(text)




if __name__ == '__main__':
    corpus = [
        "The elephant sneezed at the sight of potatoes.",
        "Bats can see via echolocation. See the bat sneeze!",
        "Wondering, she opened the door to the studio."
    ]

    test = [word for word in tokenize(corpus[0])]
    test2 = frequency_vector(corpus, tfid=True)



