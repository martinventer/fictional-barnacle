#! envs/fictional-barnacle/bin/python3.6
"""
Elsevier_Corpus_Reader.py

@author: martinventer
@date: 2019-06-10

Reads the raw data from Elsivier Ingestor and refactors it into a per article
"""

import pickle
import os
import time

from functools import partial

from sklearn.model_selection import KFold

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk import wordpunct_tokenize
import nltk

from datetime import datetime

from Utils import Utils

import logging

try:
    logging.basicConfig(filename='logs/reader.log',
                        format='%(asctime)s %(message)s',
                        level=logging.INFO)
except FileNotFoundError:
    pass


class RawCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    A reader for an entire raw corpus, as Downloaded by an ingestion engine.
    """
    PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
    CAT_PATTERN = r'([a-z_\s]+)/.*'

    def __init__(self, root, **kwargs):
        """
        Initialise the pickled corpus Pre_processor using two corpus readers
        from the nltk library
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
            kwargs['cat_pattern'] = RawCorpusReader.CAT_PATTERN

        if not any(key.startswith('pkl_') for key in kwargs.keys()):
            kwargs['pkl_pattern'] = RawCorpusReader.PKL_PATTERN

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids=kwargs['pkl_pattern'])

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
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def read_single(self, fileid=None, root=None) -> dict:
        root = self.root if (root is None) else root
        with open(os.path.join(root, fileid), 'rb') as f:
            return pickle.load(f)


class ScopusCorpusReader(RawCorpusReader):
    PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
    CAT_PATTERN = r'([a-z_\s]+)/.*'

    def __init__(self, root, **kwargs):
        """
        Initialise the pickled corpus reader using two corpus readers from
        the nltk library
        Parameters
        ----------
        root : str like
            the root directory for the corpus
        kwargs :
            Additional arguements passed to the nltk corpus readers
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = ScopusCorpusReader.CAT_PATTERN

        if not any(key.startswith('pkl_') for key in kwargs.keys()):
            kwargs['pkl_pattern'] = ScopusCorpusReader.PKL_PATTERN

        # CategorizedCorpusReader.__init__(self, kwargs)
        # CorpusReader.__init__(self, root,
        #                       fileids=ScopusCorpusReader.PKL_PATTERN)
        RawCorpusReader.__init__(self, root=root, **kwargs)

    # restructure reader with nested functions

    ## document attributes
    # 'affiliation'
    # 'prism:doi'
    # 'prism:issn'
    # 'prism:url'
    # 'pubmed-id'
    # 'source-id'
    # 'link'
    # 'eid'
    # 'dc:identifier'

    # 'prism:issueIdentifier'
    # 'prism:pageRange'
    # 'prism:publicationName'
    # 'prism:volume'
    # 'subtype'
    # 'subtypeDescription'

    # 'prism:aggregationType'
    # 'prism:coverDate'
    # 'prism:coverDisplayDate'
    # 'authkeywords'

    ## author information
    # 'author'


    ## statistics
    # 'citedby-count'
    # 'author-count'

    # 'dc:creator'
    'dc:description'
    'dc:title'

    # 'fund-no'

    # 'openaccess'
    # 'openaccessFlag'

    def affiliation_l(self, **kwargs) -> list:
        """
        Generator for document affiliations
        Parameters
        ----------

        Returns
        -------
            document affiliations

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['affiliation']
            except KeyError:
                yield []

    def affiliation_city_l(self, **kwargs) -> list:
        """
        Generator for document city affiliation list
        Parameters
        ----------

        Returns
        -------
            document city affiliation list

        """
        for affiliation in self.affiliation_l(**kwargs):
            try:
                cities =[]
                for affiliate in affiliation:
                    cities.append(affiliate['affiliation-city'])
                yield cities
            except KeyError:
                yield []

    def affiliation_city_s(self, **kwargs) -> str:
        """
        Generator for document city affiliation
        Parameters
        ----------

        Returns
        -------
            document city affiliation

        """
        for cities in self.affiliation_city_l(**kwargs):
            for city in cities:
                try:
                    yield city
                except KeyError:
                    yield ''

    def affiliation_country_l(self, **kwargs) -> list:
        """
        Generator for document county affiliation list
        Parameters
        ----------

        Returns
        -------
            document county affiliation list

        """
        for affiliation in self.affiliation_l(**kwargs):
            try:
                countries =[]
                for affiliate in affiliation:
                    countries.append(affiliate['affiliation-country'])
                yield countries
            except KeyError:
                yield []

    def affiliation_country_s(self, **kwargs) -> str:
        """
        Generator for document country affiliation
        Parameters
        ----------

        Returns
        -------
            document country affiliation

        """
        for countries in self.affiliation_country_l(**kwargs):
            for country in countries:
                try:
                    yield country
                except KeyError:
                    yield ''

    def affiliation_url_l(self, **kwargs) -> list:
        """
        Generator for document affiliation url list
        Parameters
        ----------

        Returns
        -------
            document affiliation url list

        """
        for affiliation in self.affiliation_l(**kwargs):
            try:
                urls = []
                for affiliate in affiliation:
                    urls.append(affiliate['affiliation-url'])
                yield urls
            except KeyError:
                yield []

    def affiliation_url_s(self, **kwargs) -> str:
        """
        Generator for document affiliation url
        Parameters
        ----------

        Returns
        -------
            document affiliation url

        """
        for urls in self.affiliation_url_l(**kwargs):
            for url in urls:
                try:
                    yield url
                except KeyError:
                    yield ''

    def affiliation_name_l(self, **kwargs) -> list:
        """
        Generator for document affiliation name list
        Parameters
        ----------

        Returns
        -------
            document affiliation name list

        """
        for affiliation in self.affiliation_l(**kwargs):
            try:
                names = []
                for affiliate in affiliation:
                    names.append(affiliate['affilname'])
                yield names
            except KeyError:
                yield []

    def affiliation_name_s(self, **kwargs) -> str:
        """
        Generator for document affiliation name
        Parameters
        ----------

        Returns
        -------
            document affiliation name

        """
        for names in self.affiliation_name_l(**kwargs):
            for name in names:
                try:
                    yield name
                except KeyError:
                    yield ''

    def affiliation_id_l(self, **kwargs) -> list:
        """
        Generator for document affiliation id list
        Parameters
        ----------

        Returns
        -------
            document affiliation id list

        """
        for affiliation in self.affiliation_l(**kwargs):
            try:
                ids = []
                for affiliate in affiliation:
                    ids.append(affiliate['afid'])
                yield ids
            except KeyError:
                yield []

    def affiliation_id_s(self, **kwargs) -> str:
        """
        Generator for document affiliation id
        Parameters
        ----------

        Returns
        -------
            document affiliation id

        """
        for ids in self.affiliation_id_l(**kwargs):
            for ident in ids:
                try:
                    yield ident
                except KeyError:
                    yield ''

    def keywords_l(self, **kwargs) -> list:
        """
        Generator for document author assigned keywords for document
        Parameters
        ----------

        Returns
        -------
            author keywords

        """
        for doc in self.docs(**kwargs):
            try:
                yield [keyword.strip() for keyword in doc[
                    'authkeywords'].split("|")]
            except KeyError:
                yield []

    def keywords_string(self, **kwargs) -> str:
        """
        Generator for document author assigned keywords for document as single
        string
        Parameters
        ----------

        Returns
        -------
            author keywords

        """
        for keywords in self.keywords_l(**kwargs):
            try:
                yield ' '.join(keywords)
            except KeyError:
                yield ''

    def keywords_phrase(self, **kwargs) -> str:
        """
        Generator for document author assigned keyword phrases
        Parameters
        ----------

        Returns
        -------
            author keyword phrases

        """
        for keywords in self.keywords_l(**kwargs):
            try:
                for phrase in keywords:
                    yield phrase
            except KeyError:
                yield ''

    def keywords_s(self, **kwargs) -> str:
        """
        Generator for document author assigned keyword
        Parameters
        ----------

        Returns
        -------
            author keyword

        """
        for phrase in self.keywords_phrase(**kwargs):
            words = [x.strip() for x in phrase.split(' ')]
            try:
                for word in words:
                    yield word
            except KeyError:
                yield ''

    def author_data_l(self, **kwargs) -> list:
        """
        Generator for document author data.
        Parameters
        ----------

        Returns
        -------
            list of author data

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['author']
            except KeyError:
                yield []

    def author_data_id_l(self, **kwargs) -> list:
        """
        Generator for document author ids.
        Parameters
        ----------

        Returns
        -------
            list of author ids

        """
        for data in self.author_data_l(**kwargs):
            try:
                author_ids = []
                for author in data:
                    author_ids.append(author['authid'])
                yield author_ids
            except KeyError:
                yield []

    def author_data_id_s(self, **kwargs) -> str:
        """
        Generator for document author id.
        Parameters
        ----------

        Returns
        -------
            list of author id

        """
        for identifiers in self.author_data_id_l(**kwargs):
            for identifier in identifiers:
                try:
                    yield identifier
                except KeyError:
                    yield ''

    def author_data_name_full_l(self, **kwargs) -> list:
        """
        Generator for document author full names.
        Parameters
        ----------

        Returns
        -------
            list of author full name list

        """
        for data in self.author_data_l(**kwargs):
            try:
                author_names = []
                for author in data:
                    author_names.append(author['authname'])
                yield author_names
            except KeyError:
                yield []

    def author_data_name_full_s(self, **kwargs) -> str:
        """
        Generator for document author full name.
        Parameters
        ----------

        Returns
        -------
            list of author full name

        """
        for full_names in self.author_data_name_full_l(**kwargs):
            for full_name in full_names:
                try:
                    yield full_name
                except KeyError:
                    yield ''

    def author_data_url_l(self, **kwargs) -> list:
        """
        Generator for document author urls.
        Parameters
        ----------

        Returns
        -------
            list of author url list

        """
        for data in self.author_data_l(**kwargs):
            try:
                author_urls = []
                for author in data:
                    author_urls.append(author['author-url'])
                yield author_urls
            except KeyError:
                yield []

    def author_data_url_s(self, **kwargs) -> str:
        """
        Generator for document author url.
        Parameters
        ----------

        Returns
        -------
            list of author url

        """
        for urls in self.author_data_url_l(**kwargs):
            for url in urls:
                try:
                    yield url
                except KeyError:
                    yield ''

    def author_data_name_given_l(self, **kwargs) -> list:
        """
        Generator for document author given names.
        Parameters
        ----------

        Returns
        -------
            list of author given name list

        """
        for data in self.author_data_l(**kwargs):
            try:
                author_names = []
                for author in data:
                    author_names.append(author['given-name'])
                yield author_names
            except KeyError:
                yield []

    def author_data_name_given_s(self, **kwargs) -> str:
        """
        Generator for document author given name.
        Parameters
        ----------

        Returns
        -------
            list of author given name

        """
        for names in self.author_data_name_given_l(**kwargs):
            for name in names:
                try:
                    yield name
                except KeyError:
                    yield ''

    def author_data_initial_l(self, **kwargs) -> list:
        """
        Generator for document author initials.
        Parameters
        ----------

        Returns
        -------
            list of author initials list

        """
        for data in self.author_data_l(**kwargs):
            try:
                initials = []
                for author in data:
                    initials.append(author['initials'])
                yield initials
            except KeyError:
                yield []

    def author_data_initial_s(self, **kwargs) -> str:
        """
        Generator for document author initial.
        Parameters
        ----------

        Returns
        -------
            list of author initial

        """
        for initials in self.author_data_initial_l(**kwargs):
            for initial in initials:
                try:
                    yield initial
                except KeyError:
                    yield ''

    def author_data_name_surname_l(self, **kwargs) -> list:
        """
        Generator for document author surname .
        Parameters
        ----------

        Returns
        -------
            list of author surname  list

        """
        for data in self.author_data_l(**kwargs):
            try:
                author_names = []
                for author in data:
                    author_names.append(author['surname'])
                yield author_names
            except KeyError:
                yield []

    def author_data_name_surname_s(self, **kwargs) -> str:
        """
        Generator for document author surname .
        Parameters
        ----------

        Returns
        -------
            list of author surname

        """
        for names in self.author_data_name_surname_l(**kwargs):
            for name in names:
                try:
                    yield name
                except KeyError:
                    yield ''

    def stat_num_authors(self, **kwargs) -> int:
        """
        generator for document author count
        Parameters
        ----------

        Returns
        -------
            yields the number of authors for a document

        """
        for doc in self.docs(**kwargs):
            try:
                yield int(doc['author-count']["$"])
            except (KeyError, TypeError):
                yield 0

    def stat_num_citations(self, **kwargs) -> int:
        """
        generator for number of citations for document
        Parameters
        ----------

        Returns
        -------
            number of citations for document

        """
        for doc in self.docs(**kwargs):
            try:
                yield int(doc['citedby-count'])
            except (KeyError, TypeError):
                yield 0

    def identifier_scopus(self, **kwargs) -> str:
        """
        Generator for scopus identifiers
        Parameters
        ----------

        Returns
        -------
            scopus identifier

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['dc:identifier']
            except (KeyError, TypeError):
                yield ''

    def identifier_electronic(self, **kwargs) -> str:
        """
        Generator for electronic identifiers
        Parameters
        ----------

        Returns
        -------
            electronic identifier

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['eid']
            except (KeyError, TypeError):
                yield ''

    def identifier_link(self, **kwargs) -> str:
        """
        Generator for web link identifiers
        Parameters
        ----------

        Returns
        -------
            web link identifier

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['link'][0]['@href']
            except (KeyError, TypeError):
                yield ''

    def identifier_doi(self, **kwargs) -> str:
        """
        Generator for doi identifiers
        Parameters
        ----------

        Returns
        -------
            doi identifier

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['prism:doi']
            except (KeyError, TypeError):
                yield ''

    def identifier_issn(self, **kwargs) -> str:
        """
        Generator for issn identifiers
        Parameters
        ----------

        Returns
        -------
            issn identifier

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['prism:issn']
            except (KeyError, TypeError):
                yield ''

    def identifier_pubmed(self, **kwargs) -> str:
        """
        Generator for issn identifiers
        Parameters
        ----------

        Returns
        -------
            issn identifier

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['pubmed-id']
            except (KeyError, TypeError):
                yield ''

    def identifier_source(self, **kwargs) -> str:
        """
        Generator for issn identifiers
        Parameters
        ----------

        Returns
        -------
            issn identifier

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['source-id']
            except (KeyError, TypeError):
                yield ''

    def publication_type(self, **kwargs) -> str:
        """
        Generator for publication type
        Parameters
        ----------

        Returns
        -------
            publication type

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['prism:aggregationType']
            except (KeyError, TypeError):
                yield ''

    def publication_name(self, **kwargs) -> str:
        """
        Generator for publication name
        Parameters
        ----------

        Returns
        -------
            publication name

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['prism:publicationName']
            except (KeyError, TypeError):
                yield ''

    def publication_subtype(self, **kwargs) -> str:
        """
        Generator for publication subtype
        Parameters
        ----------

        Returns
        -------
            publication subtype

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['subtypeDescription']
            except (KeyError, TypeError):
                yield ''

    def publication_volume(self, **kwargs) -> int:
        """
        Generator for publication volume number
        Parameters
        ----------

        Returns
        -------
            publication volume number

        """
        for doc in self.docs(**kwargs):
            try:
                yield int(doc['prism:volume'])
            except (KeyError, TypeError):
                yield 0

    def publication_issue(self, **kwargs) -> int:
        """
        Generator for publication issue number
        Parameters
        ----------

        Returns
        -------
            publication issue number

        """
        for doc in self.docs(**kwargs):
            try:
                yield int(doc['prism:issueIdentifier'])
            except (KeyError, TypeError):
                yield 0

    def publication_pages(self, **kwargs) -> (int, int):
        """
        Generator for publication page numbers
        Parameters
        ----------

        Returns
        -------
            publication page numbers

        """
        for doc in self.docs(**kwargs):
            try:
                yield tuple(int(p) for p in doc['prism:pageRange'].split('-'))
            except (KeyError, AttributeError):
                yield (0, 0)

    def publication_date(self, **kwargs) -> object:
        """
        Generator for publication cover date
        Parameters
        ----------

        Returns
        -------
            publication cover date

        """
        for doc in self.docs(**kwargs):
            try:
                date_string = doc['prism:coverDate']
                yield datetime.strptime(date_string, '%Y-%m-%d')
            except KeyError:
                date_string = '1800-01-01'
                yield datetime.strptime(date_string, '%Y-%m-%d')

    def publication_year(self, **kwargs) -> int:
        """
        Generator for publication cover date
        Parameters
        ----------

        Returns
        -------
            publication cover date

        """
        for date in self.publication_date(**kwargs):
            try:
                yield date.year
            except KeyError:
                date_string = '1800-01-01'
                datetime.strptime(date_string, '%Y-%m-%d')
                yield date.year

    def document_title(self, **kwargs) -> str:
        """
        generates the title of the next document in the corpus
        Parameters
        ----------

        Returns
        -------
            yields a string containing the next document title

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['dc:title']
            except KeyError:
                yield ''

    def document_description(self, **kwargs) -> str:
        """
        generates the description of the next document in the corpus
        Parameters
        ----------

        Returns
        -------
            yields a string containing the next document description

        """
        for doc in self.docs(**kwargs):
            try:
                yield doc['dc:description']
            except KeyError:
                yield ''


class ScopusProcessedCorpusReader(ScopusCorpusReader):
    PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
    CAT_PATTERN = r'([a-z_\s]+)/.*'

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
        ScopusCorpusReader.__init__(self, root, fileids, **kwargs)

    def title_tagged(self, fileids=None, categories=None) -> list:
        """
        Yields the next title as a list of sentances that are a list of
        taggged words
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------

        """
        for doc in self.docs(fileids, categories):
            try:
                yield doc["struct:title"]
            except (KeyError, TypeError):
                pass

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
            except (KeyError, TypeError):
                pass

    def title_tagged_word(self, fileids=None, categories=None) -> (str, str):
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
        'Robots'
        """
        for tagged in self.title_tagged_word(fileids, categories):
            try:
                yield tagged[0]
            except KeyError:
                yield ''

    def describe(self, fileids=None, categories=None) -> dict:

        started = time.time()

        # Structures to perform counting.
        counts = nltk.FreqDist()
        tokens = nltk.FreqDist()

        # Perform single pass over paragraphs, tokenize and count
        for title in self.title_raw(fileids, categories):
            counts['titles'] += 1

            for word in wordpunct_tokenize(title):
                counts['words'] += 1
                tokens[word] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics  = len(self.categories(self.resolve(fileids, categories)))

        # Return data structure with information
        return {
            'files':  n_fileids,
            'topics': n_topics,
            'titles':  counts['titles'],
            'words':  counts['words'],
            'vocab':  len(tokens),
            'lexdiv': float(counts['words']) / float(len(tokens)),
            'tpdoc':  float(counts['titles']) / float(n_fileids),
            'wptit':  float(counts['words']) / float(counts['titles']),
            'secs':   time.time() - started,
        }

    def ngrams(self, n=2, fileids=None, categories=None) -> tuple:
        """
        a ngram generator for the scopus corpus
        Parameters
        ----------
        n : int
            the number of words in the n-gram
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            tuple

        """
        LPAD_SYMBOL = "<s>"
        RPAD_SYMBOL = "</s>"
        nltk_ngrams = partial(
            nltk.ngrams,
            pad_right=True, right_pad_symbol=RPAD_SYMBOL,
            pad_left=True, left_pad_symbol=LPAD_SYMBOL
        )
        for sent in self.title_sents(fileids=fileids, categories=categories):
            tokens, _ = zip(*sent)
            for ngram in nltk_ngrams(tokens, n):
                yield ngram



class CorpusLoader(object):
    """
    does not work with sklearn 0.21


    A wrapper fo a corpus that can split the data into k folds.This is a neat way
    of dealing with large corpus, because the loader will return only a piece
    of the corpus.
    """
    def __init__(self, corpus, folds=None, shuffle=True):
        self.n_docs = len(corpus.fileids())
        self.corpus = corpus
        self.folds = folds

        if folds is not None:
            # Generate the KFold cross validation for the loader.
            kf = KFold(n_splits=folds, shuffle=shuffle)
            self.folds = kf.splits()

    @property
    def n_folds(self):
        """
        Returns the number of folds if it exists; 0 otherwise.
        """
        if self.folds is None: return 0
        return self.folds.n_folds

    def fileids(self, fold=None, train=False, test=False):

        if fold is None:
            # If no fold is specified, return all the fileids.
            return self.corpus.fileids()

        # Otherwise, identify the fold specifically and get the train/test idx
        train_idx, test_idx = [split for split in self.folds][fold]

        # Now determine if we're in train or test mode.
        if not (test or train) or (test and train):
            raise ValueError(
                "Please specify either train or test flag"
            )

        # Select only the indices to filter upon.
        indices = train_idx if train else test_idx
        return [
            fileid for doc_idx, fileid in enumerate(self.corpus.fileids())
            if doc_idx in indices
        ]

    def documents(self, fold=None, train=False, test=False):
        for fileid in self.fileids(fold, train, test):
            yield list(self.corpus.docs(fileids=fileid))

    def titles(self, fold=None, train=False, test=False):
        for fileid in self.fileids(fold, train, test):
            yield list(self.corpus.title_tagged(fileids=fileid))

    def labels(self, fold=None, train=False, test=False):
        return [
            self.corpus.categories(fileids=fileid)[0]
            for fileid in self.fileids(fold, train, test)
        ]


class CorpuKfoldLoader(object):
    """
    A wrapper for a corpus that splits the corpus using k-fold method.
    """
    def __init__(self, corpus, n_folds=None, shuffle=True):
        self.n_folds = len(corpus.fileids())
        self.corpus = corpus

        if n_folds is not None:
            # Generate the KFold cross validation for the loader.
            kf = KFold(n_splits=n_folds, shuffle=shuffle)
            self.folds = kf.split(corpus.fileids())

    def fileids(self, train=False, test=False):

        if self.n_folds is None:
            # If no fold is specified, return all the fileids.
            return self.corpus.fileids()

        # determine if we're in train or test mode.
        if not (test or train) or (test and train):
            raise ValueError(
                "Please specify either train or test flag"
            )

        # get the next test and train set
        for train_idx, test_idx in self.folds:

            # Select only the indices to filter upon.
            indices = train_idx if train else test_idx
            yield [
                fileid for doc_idx, fileid in enumerate(self.corpus.fileids())
                if doc_idx in indices
            ]


if __name__ == '__main__':
    from pprint import PrettyPrinter
    # RawCorpusReader
    # root = "Corpus/Split_corpus/"
    # root = "Corpus/Raw_corpus/"
    # corpus = RawCorpusReader(root=root)

    # ScopusCorpusReader
    root = "Corpus/Split_corpus/"
    corpus = ScopusCorpusReader(root=root)
    # gen = corpus.affiliation()
    gen = corpus.docs()
    # gen = corpus.author_keywords_l()
    aa = next(gen)
    pp = PrettyPrinter(indent=4)
    pp.pprint(aa)
