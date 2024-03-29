#! envs/fictional-barnacle/bin/python3.6
"""
Utils.py

@author: martinventer
@date: 2019-07-09

Miscellaneous utilities to Corpus processing
"""
import logging
import os
import unicodedata
import csv


def get_key() -> str:
    """
    Reads the API key from and external file called 'api_key.dict' in the
    local directory. The format for the file is a single line containing the
    following.

        {'ElsevierDeveloper':'XXX-key-XXX'}

    A key can be obtained by registering on http://dev.elsevier.com/ and
    requesting an api key.
    Parameters
    ----------

    Returns
    -------
        str
            api key as string
    """
    file = 'api_key.dict'
    key = 'ElsevierDeveloper'
    return eval(open(file, 'r').read())[key]


def make_folder(path) -> None:
    """
    creates a directory without failing unexpectedly
    Parameters
    ----------
    path : str
        a string containing the desired path

    Returns
    -------
        None
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        logging.info("file %s already exists" % path)
        pass


def iter_flatten(iterable):
    """
    A recursive iterator to flatten nested lists
    Parameters
    ----------
    iterable

    Returns
    -------
        yields next item
            to get a flat the nested list 'a'
                a = [i for i in iter_flatten(a)]
    """
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in iter_flatten(e):
                yield f
        else:
            yield e


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


def academic_stopwords(existing_list=None) -> list:
    with open('Utils/stopwords.csv', 'r') as f:
        reader = csv.reader(f)
        word_list = list(reader)

    flat_word_list = [i for i in iter_flatten(word_list)]

    if existing_list:
        flat_word_list = flat_word_list + list(existing_list)

    return flat_word_list


if __name__ == '__main__':
    print(academic_stopwords())