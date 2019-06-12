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

from datetime import datetime

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
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            a list of file Ids for the corpus
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None) -> dict:
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. Similar to the BaleenCorpusReader, this uses a generator
        to archeive memory safe iteration.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a dictionary containing all the metadata for a given document

        Example output
        --------------
        {
        '@_fa': 'true',
        'load-date': '2003-07-30T00:00:00Z',
        'link': [
            {
                '@_fa': 'true',
                '@ref': 'self',
                '@href': 'https://api.elsevier.com/content/article/pii/0167713686900351'
            },
            {
                '@_fa': 'true',
                '@ref': 'scidir',
                '@href': 'https://www.sciencedirect.com/science/article/pii/0167713686900351?dgcid=api_sd_search-api-endpoint'
            }
                ],
        'dc:identifier': 'DOI:10.1016/0167-7136(86)90035-1',
        'prism:url': 'https://api.elsevier.com/content/article/pii/0167713686900351',
        'dc:title': 'Knowledge resource tools for information access',
        'dc:creator': 'D. E. Walker',
        'prism:publicationName': 'Computer Compacts',
        'prism:volume': '4',
        'prism:coverDate': '1986-10-31',
        'prism:startingPage': '182',
        'prism:doi': '10.1016/0167-7136(86)90035-1',
        'openaccess': False,
        'pii': '0167713686900351',
        'author_list':  {
            'author': 'D. E. Walker'
                    }
        }


        """
        fileids = self.resolve(fileids, categories)
        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def titles(self, fileids=None, categories=None) -> str:
        """
        generates the title of the next document in the corpus
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next document title
        or
            'NO TITLE'

        Example output
        --------------
        'Knowledge resource tools for information access'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['dc:title']
            except KeyError:
                yield 'NO TITLE'

    def title_words(self, fileids=None, categories=None) -> str:
        """
        generates the next title word in the corpus
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next title word
        or
            'NO WORD'

        Example output
        --------------
        'Knowledge'
        """
        for doc in self.docs(fileids, categories):
            try:
                for word in wordpunct_tokenize(doc['dc:title']):
                    yield word
            except KeyError:
                yield 'NO WORD'

    def doc_ids(self, fileids=None, categories=None) -> str:
        """
        generates the next document in the corpus. Typically a DOI Number or
        similar
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next document ID

        or
            'NO DOC ID'

        Example output
        --------------
        'DOI:10.1016/0167-7136(86)90035-1'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['dc:identifier']
            except KeyError:
                yield 'NO DOC ID'

    def publication(self, fileids=None, categories=None) -> str:
        """
        generates the next journal or publication name in the corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next publication name

        or
            'NO PUBLICATION NAME'

        Example output
        --------------
        'Computer Compacts'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['prism:publicationName']
            except KeyError:
                yield 'NO PUBLICATION NAME'

    def pub_date(self, fileids=None, categories=None, form=None) -> object:
        """
        generates the next date of publication in the corpus.
        Parameters
        ----------
        form: str
            'year' - restricts output to year of publication only
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a datetime object containing the next date of publication

        or
            'NO PUB DATE'

        Example output
        --------------
        1986-10-31
        """
        for doc in self.docs(fileids, categories):
            try:
                date_string = doc['prism:coverDate']
                if form is None:
                    yield datetime.strptime(date_string, '%Y-%m-%d')
                elif form is 'year':
                    yield datetime.strptime(date_string, '%Y-%m-%d').year
            except KeyError:
                yield None

    def pub_type(self, fileids=None, categories=None) -> str:
        """
        generates the next document type in the corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next  document type

        or
            'NO PUB TYPE'

        Example output
        --------------
        'Article'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['subtypeDescription']
            except KeyError:
                yield 'NO PUB TYPE'

    def author_list(self, fileids=None, categories=None) -> dict:
        """
        generates the dictionary of author_list for the next in the corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a dictionary containing the author_list of the  next document
            type

        or
            None

        Example output
        --------------
        {'author':  [
            {'$': 'H. Aiso'},
            {'$': 'F. Kuo'},
            {'$': 'R. P. van de Riet'}
                    ]
        }
        or
        {'author': 'Lung-Sing Liang'}
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['authors']
            except KeyError:
                yield None


    def author_count(self, fileids=None, categories=None) -> str:
        """
        generates the number of authors in the next document in the corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields the number of authors in the next document

        or
            None

        Example output
        --------------
        1
        """
        for authors in self.author_list(fileids, categories):
            try:
                if type(authors['author']) is str:
                    yield 1
                elif type(authors['author']) is list:
                    yield len(authors['author'])
                elif type(authors['author']) is NoneType:
                    yield 0
            except (KeyError, TypeError):
                yield None


    def author(self, fileids=None, categories=None) -> str:
        """
        generates the an author next in the corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields the next author

        or
            None

        Example output
        --------------
        'D. E. Walker'
        """
        for authors in self.author_list(fileids, categories):
            try:
                if type(authors['author']) is str:
                    yield authors['author']
                elif type(authors['author']) is list:
                    for s in authors['author']:
                        yield s['$']
            except (KeyError, TypeError):
                yield None

    def author_keywords(self, fileids=None, categories=None) -> str:
        """
        generates a string of author keywords for the next document in the
        corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next  author keywords

        or
            'NO KEYWORDS'

        Example output
        --------------
        'EGaIn | Liquid metal | Lorentz force | Mixer | Self-rotation'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['authkeywords']
            except KeyError:
                yield 'NO KEYWORDS'

    def doc_url(self, fileids=None, categories=None) -> str:
        """
        generates a string of document URL for the next document in the
        corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next document URL

        or
            'NO URL'

        Example output
        --------------
        'https://api.elsevier.com/content/article/pii/0167713686900351'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['prism:url']
            except KeyError:
                yield 'NO URL'

    def doc_volume(self, fileids=None, categories=None) -> int:
        """
        generates an intiger number for the volume of the next document in the
        corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields an intiger number for the volume

        or
            None

        Example output
        --------------
        4
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['prism:volume']
            except KeyError:
                yield None

    def doc_first_page(self, fileids=None, categories=None) -> int:
        """
        generates an intiger number for the first page number of the next
        document in the corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields an intiger number for the first page number

        or
            None

        Example output
        --------------
        132
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['prism:startingPage']
            except KeyError:
                yield None

    def doc_doi(self, fileids=None, categories=None) -> str:
        """
        generates a string of document DOI for the next document in the
        corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next document DOI

        or
            'NO DOI'

        Example output
        --------------
        '10.1016/0167-7136(86)90035-1'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['prism:doi']
            except KeyError:
                yield 'NO DOI'

    def doc_pii(self, fileids=None, categories=None) -> str:
        """
        generates a string of document PII for the next document in the
        corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next document PII

        or
            'NO PII'

        Example output
        --------------
        '0167713686900351'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['pii']
            except KeyError:
                yield 'NO PII'

    def read_single(self, fileid=None, root=None):
        """
        Depricated single read. with new method of preprocessing data,
        each pickle file contains only one file.
        Parameters
        ----------
        fileid: basestring
            Name of file name
        root: basestring
            Root directoy

        Returns
        -------
            a dictionary object containing a the meta-data for a single article
        """
        root = self.root if (root is None) else root
        print(root)
        with open(os.path.join(root, fileid), 'rb') as f:
            return pickle.load(f)
