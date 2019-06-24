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


class ScopusRawCorpusReader(CategorizedCorpusReader, CorpusReader):

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
        {'@_fa': 'true',
         'link': [{'@_fa': 'true',
           '@ref': 'self',
           '@href': 'https://api.elsevier.com/content/abstract/scopus_id/85062801216'},
          {'@_fa': 'true',
           '@ref': 'author_name-affiliation',
           '@href': 'https://api.elsevier.com/content/abstract/scopus_id/85062801216?field=author,affiliation'},
          {'@_fa': 'true',
           '@ref': 'scopus',
           '@href': 'https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp=85062801216&origin=inward'},
          {'@_fa': 'true',
           '@ref': 'scopus-citedby',
           '@href': 'https://www.scopus.com/inward/citedby.uri?partnerID=HzOxMe3b&scp=85062801216&origin=inward'}],
         'prism:url': 'https://api.elsevier.com/content/abstract/scopus_id/85062801216',
         'dc:identifier': 'SCOPUS_ID:85062801216',
         'eid': '2-s2.0-85062801216',
         'dc:title': 'Spatio-Temporal Reasoning within a Neural Network framework for Intelligent Physical Systems',
         'dc:creator': 'Sathish Kumar A.',
         'prism:publicationName': 'Proceedings of the 2018 IEEE Symposium Series on Computational Intelligence, SSCI 2018',
         'prism:isbn': [{'@_fa': 'true', '$': '9781538692769'}],
         'prism:pageRange': '274-280',
         'prism:coverDate': '2019-01-28',
         'prism:coverDisplayDate': '28 January 2019',
         'prism:doi': '10.1109/SSCI.2018.8628748',
         'dc:description': '© 2018 IEEE. Existing functionality for intelligent physical systems (IPS), such as autonomous vehicles (AV), generally lacks the ability to reason and evaluate the environment and to learn from other intelligent agents in an autonomous fashion. Such capabilities for IPS is required for scenarios where an human intervention is unlikely to be available and robust long-term autonomous operation is necessary in potentially dynamic environments. To address these issues, the IPS will then need to reason about the interactions with these items through time and space. Incorporating spatio-temporal reasoning into the IPS will provide the capability to understand these interactions. This paper describes our proposed neural network framework that incorporates spatio-temporal reasoning for IPS. The preliminary experimental results addressing research challenges related to spatio-temporal reasoning within neural network framework for IPS are promising.',
         'citedby-count': '0',
         'affiliation': [{'@_fa': 'true',
           'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60019213',
           'afid': '60019213',
           'affilname': 'Coastal Carolina University',
           'affiliation-city': 'Conway',
           'affiliation-country': 'United States'},
          {'@_fa': 'true',
           'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/101522664',
           'afid': '101522664',
           'affilname': 'Research and Development Service',
           'affiliation-city': 'San Antonio',
           'affiliation-country': 'United States'}],
         'prism:aggregationType': 'Conference Proceeding',
         'subtype': 'cp',
         'subtypeDescription': 'Conference Paper',
         'author_name-count': {'@limit': '100', '$': '2'},
         'author_name': [{'@_fa': 'true',
           '@seq': '1',
           'author_name-url': 'https://api.elsevier.com/content/author/author_id/57195136226',
           'authid': '57195136226',
           'authname': 'Sathish Kumar A.',
           'surname': 'Sathish Kumar',
           'given-name': 'A. P.',
           'initials': 'A.P.',
           'afid': [{'@_fa': 'true', '$': '60019213'}]},
          {'@_fa': 'true',
           '@seq': '2',
           'author_name-url': 'https://api.elsevier.com/content/author/author_id/57207867215',
           'authid': '57207867215',
           'authname': 'Brown M.',
           'surname': 'Brown',
           'given-name': 'Michael A.',
           'initials': 'M.A.',
           'afid': [{'@_fa': 'true', '$': '101522664'}]}],
         'authkeywords': 'Automated Vehicles | convolution neural networks | Spatio-Temporal Reasoning',
         'article-number': '8628748',
         'source-id': '21100901193',
         'fund-no': 'undefined',
         'openaccess': '0',
         'openaccessFlag': False}
        """
        fileids = self.resolve(fileids, categories)
        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def title_raw(self, fileids=None, categories=None) -> str:
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
            ''

        Example output
        --------------
        'Knowledge resource tools for information access'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['dc:title']
            except KeyError:
                yield ''

    def abstracts(self, fileids=None, categories=None) -> str:
        """
        generates the abstracts of the next document in the corpus
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
            ''

        Example output
        --------------
        'Knowledge resource tools for information access'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['dc:description']
            except KeyError:
                yield ''

    def doc_ids(self, fileids=None, categories=None, form='prism:url') -> str:
        """
        generates the next document in the corpus. Typically a DOI Number or
        similar
        Parameters
        ----------
            form: str default 'prism:url'
                form of document Identification
                    'prism:url' - Content Abstract Retrieval API URI
                    'dc:identifier' - Scopus ID
                    'eid' - Electronic ID
                    'prism:isbn' - Source Identifier
                    'prism:doi' - Document Object Identifier
                    'article-number' - Article Number

            fileids: basestring or None
                complete path to specified file
            categories: basestring or None
                path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next document ID

        or
            ''

        Example output
        --------------
        'SCOPUS_ID:85062801216'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc[form]
            except KeyError:
                yield ''

    def publication(self, fileids=None, categories=None,
                    form='prism:publicationName') -> str:
        """
        generates the next journal or publication name in the corpus.
        Parameters
        ----------
        form: str default 'prism:publicationName'
                form of document Identification
                    'prism:publicationName' - Source Title
                    'subtypeDescription' - Document Type description
                    'prism:aggregationType' - Source Type
                    'subtype' - Document Type code

        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a string containing the next publication name

        or
            ''

        Example output
        --------------
        'Computer Compacts'
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc[form]
            except KeyError:
                yield ''

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
            None

        Example output
        --------------
        2019-01-28 00:00:00
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

    def author_data(self, fileids=None, categories=None) -> list:
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
            [{'@_fa': 'true',
              '@seq': '1',
              'author_name-url': 'https://api.elsevier.com/content/author/author_id/57207938844',
              'authid': '57207938844',
              'authname': 'Tamatsukuri A.',
              'surname': 'Tamatsukuri',
              'given-name': 'Akihiro',
              'initials': 'A.',
              'afid': [{'@_fa': 'true', '$': '60003414'}]},
             {'@_fa': 'true',
              '@seq': '2',
              'author_name-url': 'https://api.elsevier.com/content/author/author_id/7406460920',
              'authid': '7406460920',
              'authname': 'Takahashi T.',
              'surname': 'Takahashi',
              'given-name': 'Tatsuji',
              'initials': 'T.',
              'afid': [{'@_fa': 'true', '$': '60003414'},
               {'@_fa': 'true', '$': '116598425'}]}]
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['author']
            except KeyError:
                yield []

    def author_count(self, fileids=None, categories=None) -> int:
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
        for doc in self.docs(fileids, categories):
            try:
                yield int(doc['author-count']["$"])
            except (KeyError, TypeError):
                yield None

    def author_name(self, fileids=None, categories=None, form='authname') -> \
            str:
        """
        generates the an author next in the corpus.
        Parameters
        ----------
            form : str default 'authname'
                form of the author name requested options:
                    'author-url' - author scopus URL
                    'authid' - scopus author ID
                    'authname' - full author name and initial
                    'surname' - author surname only
                    'given-name' - author given name only
                    'initials' - initials
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
        for authors in self.author_data(fileids, categories):
            try:
                if type(authors) is list:
                    for author in authors:
                        yield author[form]
            except (KeyError, TypeError):
                yield None

    def author_list(self, fileids=None, categories=None, form='authname') -> \
            list:
        """
        generates the a list of authors for a document next in the corpus.
        Parameters
        ----------
            form : str default 'authname'
                form of the author name requested options:
                    'author-url' - author scopus URL
                    'authid' - scopus author ID
                    'authname' - full author name and initial
                    'surname' - author surname only
                    'given-name' - author given name only
                    'initials' - initials
            fileids: basestring or None
                complete path to specified file
            categories: basestring or None
                path to directory containing a subset of the fileids

        Returns
        -------
            yields list of authors on a document

        or
            []

        Example output
        --------------
        ['D. E. Walker', 'Tamatsukuri A.']
        """
        for authors in self.author_data(fileids, categories):
            try:
                if type(authors) is list:
                    yield [author[form] for author in authors]
            except (KeyError, TypeError):
                yield []

    def author_keyword_list(self, fileids=None, categories=None) -> list:
        """
        generates a list of author keywords for the next document in the
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
            []

        Example output
        --------------
        ['Bio-inspired design', 'Minimally invasive', 'Paramecium']
        """
        for doc in self.docs(fileids, categories):
            try:
                yield [keyword.strip() for keyword in doc[
                    'authkeywords'].split("|")]
            except KeyError:
                yield []

    def author_keyword(self, fileids=None, categories=None) -> str:
        """
        generates a string for author keywords for the next document in the
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
            ''

        Example output
        --------------
        'Automated Vehicles'
        """
        for keywords in self.author_keyword_list(fileids, categories):
            try:
                for keyword in keywords:
                    yield keyword
            except KeyError:
                yield ''

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

    def doc_page_range(self, fileids=None, categories=None) -> (int, int):
        """
        generates an intiger tupple for the first and last page number of the
        next
        document in the corpus.
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields an intiger tuple for the first and last page number

        or
            (None, None)

        Example output
        --------------
        (274, 280)
        """
        for doc in self.docs(fileids, categories):
            try:
                yield tuple(int(p) for p in doc['prism:pageRange'].split('-'))
            except (KeyError, AttributeError):
                yield (None, None)

    def doc_citation_number(self, fileids=None, categories=None) -> int:
        """
        generator for number of citations
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            int

        or
            None

        Example output
        --------------
        4
        """
        for doc in self.docs(fileids, categories):
            try:
                yield int(doc['citedby-count']['$'])
            except KeyError:
                if type(doc['citedby-count']) is str:
                    yield int(doc['author_name-count'])
                else:
                    yield int(doc['author_name-count']['$'])
            except TypeError:
                yield int(doc['citedby-count'])

    def affiliation_list(self, fileids=None, categories=None) -> list:
        """
        generates list of affiliatons.
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
            [{'@_fa': 'true',
              'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60003414',
              'afid': '60003414',
              'affilname': 'Tokyo Denki University',
              'affiliation-city': 'Tokyo',
              'affiliation-country': 'Japan'},
             {'@_fa': 'true',
              'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/116598425',
              'afid': '116598425',
              'affilname': 'Dwango Artificial Intelligence Laboratory',
              'affiliation-city': 'Tokyo',
              'affiliation-country': 'Japan'}]
        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc['affiliation']
            except KeyError:
                yield []

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
        with open(os.path.join(root, fileid), 'rb') as f:
            return pickle.load(f)


class ScopusProcessedCorpusReader(ScopusRawCorpusReader):

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
        ScopusRawCorpusReader.__init__(self, root, fileids, **kwargs)

    def title_sents(self, fileids=None, categories=None) -> list:
        """
        Gets the next title sentence
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields a taggeed list fo tupples
        or
            []

        Example output
        --------------
        [('Robots', 'NNS'),(',', ','),('productivity', 'NN'),('and', 'CC'),
        ('quality', 'NN')]
        """
        for doc in self.docs(fileids, categories):
            try:
                for sent in doc["struct:title"]:
                    yield sent
            except KeyError:
                yield []

    def title_tagged(self, fileids=None, categories=None) -> (str, str):
        """
        yields the next tagged word in the title
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields the next tagged word in the title
        or
            ('','')

        Example output
        --------------
        ('Robots', 'NNS')
        """
        for sent in self.title_sents(fileids, categories):
            try:
                for tagged_token in sent:
                    yield tagged_token
            except KeyError:
                yield ('', '')

    def title_words(self, fileids=None, categories=None) -> str:
        """
        yields the next word in the title
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            yields the next word in the title
        or
            ''

        Example output
        --------------
        ('Robots', 'NNS')
        """
        for tagged in self.title_tagged(fileids, categories):
            try:
                yield tagged[0]
            except KeyError:
                yield ''

    def abstract_paras(self, fileids=None, categories=None) -> str:
        """
        a generator for abstract paragraphs
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            basestring
        """
        for abstract in self.abstracts(fileids, categories):
            try:
                for paragraph in abstract.split("\n"):
                    yield paragraph
            except KeyError:
                yield ''

    def abstract_sents(self, fileids=None, categories=None) -> str:
        """
        a generator for abstract sents
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            basestring
        """
        for paragraph in self.abstract_paras(fileids, categories):
            try:
                for sent in paragraph.split(". "):
                    yield sent
            except KeyError:
                yield ''

    def abstract_words(self, fileids=None, categories=None) -> str:
        """
        a generator for abstract words
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            basestring
        """
        for sent in self.abstract_sents(fileids, categories):
            try:
                for word in wordpunct_tokenize(sent):
                    yield word
            except KeyError:
                yield ""

