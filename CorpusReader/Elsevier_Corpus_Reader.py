# /home/martin/Documents/RESEARCH/fictional-barnacle/CorpusReader/
"""
Elsevier_Corpus_Reader.py

@author: martinventer
@date: 2019-06-10

Read the pickled files from the Elsivier Ingestor
"""

import pickle
import os

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

from nltk import wordpunct_tokenize

import logging


logging.basicConfig(filename='logs/reader.log',
                    format='%(asctime)s %(message)s',
                    level=logging.INFO)


# PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
# PKL_PATTERN = r'(?!\.)[0-9_\s]+/[a-f0-9]+\.pickle'
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[0-9_\s]+/[a-f0-9]+\.pickle'

# CAT_PATTERN = r'([a-z_\s]+)/.*'
# CAT_PATTERN = r'([0-9_\s]+)/.*'
CAT_PATTERN = r'([a-z_\s]+/[0-9_\s]+)/.*'


class PickledCorpusReader(CategorizedCorpusReader, CorpusReader):

    def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
        """
        Initialise the pickled corpus reader using two corpus readers from
        the nltk library
        Parameters
        ----------
        root : str like
            the root directory for the corpus
        fileids : str like
            a regex pattern for the corpus document files
        kwargs :
            Additional arguements passed to the nltk corpus readers
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids, categories):
        """
         Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. This primarily bubbles up to
        the high level ``docs`` method, but is implemented here similar to
        the nltk ``CategorizedPlaintextCorpusReader``.
        Parameters
        ----------
        fileids :
        categories :

        Returns
        -------

        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. Similar to the BaleenCorpusReader, this uses a generator
        to archeive memory safe iteration.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def titles(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            for entity in doc:
                try:
                    yield entity['dc:title']
                except:
                    yield 'NO TITLE'

    def words(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            for entity in doc:
                for word in wordpunct_tokenize(entity['dc:title']):
                    try:
                        yield word
                    except:
                        yield 'NO WORD'

    def doc_ids(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            for entity in doc:
                try:
                    yield entity['dc:identifier']
                except:
                    yield 'NO DOC ID'

    def publication(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            for entity in doc:
                try:
                    yield entity['prism:publicationName']
                except:
                    yield 'NO PUBLICATION'

    def pub_date(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            for entity in doc:
                try:
                    yield entity['prism:coverDate']
                except:
                    yield 'NO PUB DATE'

    def pub_type(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            for entity in doc:
                try:
                    yield entity['subtypeDescription']
                except:
                    yield 'NO PUB TYPE'

    def authors(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            for entity in doc:
                try:
                    yield entity['author']
                except:
                    yield 'NO AUTHORS'

    def author_keywords(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            for entity in doc:
                try:
                    yield entity['authkeywords']
                except:
                    yield 'NO KEYWORDS'

    def read_single(self, fileid=None, root=None):
        root = self.root if (root is None) else root
        print(root)
        with open(os.path.join(root, fileid), 'rb') as f:
            return pickle.load(f)
