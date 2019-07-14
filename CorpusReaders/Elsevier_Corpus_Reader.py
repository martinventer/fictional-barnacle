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
from collections import defaultdict

from sklearn.model_selection import KFold

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk import wordpunct_tokenize
import nltk

from datetime import datetime

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

    def describe(self, fileids=None, categories=None) -> dict:

        started = time.time()

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics = len(self.categories(self.resolve(fileids, categories)))

        # Return data structure with information
        return {
            'files':  n_fileids,
            'topics': n_topics,
            'secs':   time.time() - started,
        }


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

        RawCorpusReader.__init__(self, root=root, **kwargs)

    def _doc_2_dic_gen_l(self, attribute,  **kwargs) -> list:
        """
        a helper method to extract fields from raw document data

        Parameters
        ----------
        attribute : str
            the attribute to be returned
        """
        for doc in self.docs(**kwargs):
            try:
                yield doc[attribute]
            except KeyError:
                yield []

    def _doc_2_str_gen_s(self, attribute,  **kwargs) -> str:
        """
        a helper method to extract fields from raw document data

        Parameters
        ----------
        attribute : str
            the attribute to be returned
        """
        for doc in self.docs(**kwargs):
            try:
                text = doc[attribute]
            except (KeyError, TypeError):
                text = 'unk'
            if text is not None:
                yield text
            else:
                yield 'unk'

    @staticmethod
    def _dic_2_list_gen_l(gen, attribute) -> list:
        """
        a helper method to convert raw document data to list

        Parameters
        ----------
        attribute : str
            the attribute to be returned
        """
        for _dict in gen:
            try:
                yield [_item[attribute] for
                       _item in _dict]
            except KeyError:
                yield []

    @staticmethod
    def _list_2_string_gen_s(gen) -> str:
        """
        a helper method to convert raw affiliation data to list per document

        Parameters
        ----------
        gen : list
            the attribute to be returned
        """
        for _list in gen:
            if _list:
                for _item in _list:
                    if _item is not None:
                        yield _item
                    else:
                        yield 'unk'
            else:
                yield 'unk'

    def affiliation_l(self, **kwargs) -> list:
        """
        Generator for document affiliations
        Parameters
        ----------

        Returns
        -------
            document affiliations

        """
        yield from self._doc_2_dic_gen_l('affiliation',  **kwargs)

    def affiliation_city_l(self, **kwargs) -> list:
        """
        Generator for document city affiliation list. This is an as is list
        from the raw data, and may contain None values.
        Parameters
        ----------

        Returns
        -------
            document city affiliation list

        """
        yield from self._dic_2_list_gen_l(gen=self.affiliation_l(**kwargs),
                                          attribute='affiliation-city')

    def affiliation_city_s(self, **kwargs) -> str:
        """
        Generator for document city affiliation. This draws single city names
        from the city affiliation list. None values in the list are replaced
        with 'unknownCity'
        Parameters
        ----------

        Returns
        -------
            document city affiliation

        """
        yield from self._list_2_string_gen_s(
            self.affiliation_city_l(**kwargs))

    def affiliation_country_l(self, **kwargs) -> list:
        """
        Generator for document county affiliation list. list may containg
        NoneType data.
        Parameters
        ----------

        Returns
        -------
            document county affiliation list

        """
        yield from self._dic_2_list_gen_l(gen=self.affiliation_l(**kwargs),
                                          attribute='affiliation-country')

    def affiliation_country_s(self, **kwargs) -> str:
        """
        Generator for document country affiliation. None values in the list are
        replaced  with 'unknownCountry'
        Parameters
        ----------

        Returns
        -------
            document country affiliation

        """
        yield from self._list_2_string_gen_s(
            self.affiliation_country_l(**kwargs))

    def affiliation_url_l(self, **kwargs) -> list:
        """
        Generator for document affiliation url list
        Parameters
        ----------

        Returns
        -------
            document affiliation url list

        """
        yield from self._dic_2_list_gen_l(gen=self.affiliation_l(**kwargs),
                                          attribute='affiliation-url')

    def affiliation_url_s(self, **kwargs) -> str:
        """
        Generator for document affiliation url
        Parameters
        ----------

        Returns
        -------
            document affiliation url

        """
        yield from self._list_2_string_gen_s(
            self.affiliation_url_l(**kwargs))

    def affiliation_name_l(self, **kwargs) -> list:
        """
        Generator for document affiliation name list
        Parameters
        ----------

        Returns
        -------
            document affiliation name list

        """
        yield from self._dic_2_list_gen_l(gen=self.affiliation_l(**kwargs),
                                          attribute='affilname')

    def affiliation_name_s(self, **kwargs) -> str:
        """
        Generator for document affiliation name
        Parameters
        ----------

        Returns
        -------
            document affiliation name

        """
        yield from self._list_2_string_gen_s(
            self.affiliation_name_l(**kwargs))

    def affiliation_id_l(self, **kwargs) -> list:
        """
        Generator for document affiliation id list
        Parameters
        ----------

        Returns
        -------
            document affiliation id list

        """
        yield from self._dic_2_list_gen_l(gen=self.affiliation_l(**kwargs),
                                          attribute='afid')

    def affiliation_id_s(self, **kwargs) -> str:
        """
        Generator for document affiliation id
        Parameters
        ----------

        Returns
        -------
            document affiliation id

        """
        yield from self._list_2_string_gen_s(
            self.affiliation_id_l(**kwargs))

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
            if keywords:
                yield ' '.join(keywords)
            else:
                yield 'unk'

    def keywords_phrase(self, **kwargs) -> str:
        """
        Generator for document author assigned keyword phrases
        Parameters
        ----------

        Returns
        -------
            author keyword phrases

        """
        yield from self._list_2_string_gen_s(
            self.keywords_l(**kwargs))

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
                    yield word.strip()
            except KeyError:
                yield 'unk'

    def author_data_l(self, **kwargs) -> list:
        """
        Generator for document author data.
        Parameters
        ----------

        Returns
        -------
            list of author data

        """
        yield from self._doc_2_dic_gen_l('author', **kwargs)

    def author_data_id_l(self, **kwargs) -> list:
        """
        Generator for document author ids.
        Parameters
        ----------

        Returns
        -------
            list of author ids

        """
        yield from self._dic_2_list_gen_l(gen=self.author_data_l(**kwargs),
                                          attribute='authid')

    def author_data_id_s(self, **kwargs) -> str:
        """
        Generator for document author id.
        Parameters
        ----------

        Returns
        -------
            list of author id

        """
        yield from self._list_2_string_gen_s(
            self.author_data_id_l(**kwargs))

    def author_data_name_full_l(self, **kwargs) -> list:
        """
        Generator for document author full names.
        Parameters
        ----------

        Returns
        -------
            list of author full name list

        """
        yield from self._dic_2_list_gen_l(gen=self.author_data_l(**kwargs),
                                          attribute='authname')

    def author_data_name_full_s(self, **kwargs) -> str:
        """
        Generator for document author full name.
        Parameters
        ----------

        Returns
        -------
            list of author full name

        """
        yield from self._list_2_string_gen_s(
            self.author_data_name_full_l(**kwargs))

    def author_data_url_l(self, **kwargs) -> list:
        """
        Generator for document author urls.
        Parameters
        ----------

        Returns
        -------
            list of author url list

        """
        yield from self._dic_2_list_gen_l(gen=self.author_data_l(**kwargs),
                                          attribute='author-url')

    def author_data_url_s(self, **kwargs) -> str:
        """
        Generator for document author url.
        Parameters
        ----------

        Returns
        -------
            list of author url

        """
        yield from self._list_2_string_gen_s(
            self.author_data_url_l(**kwargs))

    def author_data_name_given_l(self, **kwargs) -> list:
        """
        Generator for document author given names.
        Parameters
        ----------

        Returns
        -------
            list of author given name list

        """
        yield from self._dic_2_list_gen_l(gen=self.author_data_l(**kwargs),
                                          attribute='given-name')

    def author_data_name_given_s(self, **kwargs) -> str:
        """
        Generator for document author given name.
        Parameters
        ----------

        Returns
        -------
            list of author given name

        """
        yield from self._list_2_string_gen_s(
            self.author_data_name_given_l(**kwargs))

    def author_data_initial_l(self, **kwargs) -> list:
        """
        Generator for document author initials.
        Parameters
        ----------

        Returns
        -------
            list of author initials list

        """
        yield from self._dic_2_list_gen_l(gen=self.author_data_l(**kwargs),
                                          attribute='initials')

    def author_data_initial_s(self, **kwargs) -> str:
        """
        Generator for document author initial.
        Parameters
        ----------

        Returns
        -------
            list of author initial

        """
        yield from self._list_2_string_gen_s(
            self.author_data_initial_l(**kwargs))

    def author_data_name_surname_l(self, **kwargs) -> list:
        """
        Generator for document author surname .
        Parameters
        ----------

        Returns
        -------
            list of author surname  list

        """
        yield from self._dic_2_list_gen_l(gen=self.author_data_l(**kwargs),
                                          attribute='surname')

    def author_data_name_surname_s(self, **kwargs) -> str:
        """
        Generator for document author surname .
        Parameters
        ----------

        Returns
        -------
            list of author surname

        """
        yield from self._list_2_string_gen_s(
            self.author_data_name_surname_l(**kwargs))

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
            if doc:
                try:
                    yield int(doc['author-count']["$"])
                except (KeyError, TypeError):
                    yield -1
            else:
                yield -1

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
            if doc:
                try:
                    yield int(doc['citedby-count'])
                except (KeyError, TypeError):
                    yield -1
            else:
                yield -1

    def identifier_scopus(self, **kwargs) -> str:
        """
        Generator for scopus identifiers
        Parameters
        ----------

        Returns
        -------
            scopus identifier

        """
        yield from self._doc_2_str_gen_s(attribute='dc:identifier', **kwargs)

    def identifier_electronic(self, **kwargs) -> str:
        """
        Generator for electronic identifiers
        Parameters
        ----------

        Returns
        -------
            electronic identifier

        """
        yield from self._doc_2_str_gen_s(attribute='eid', **kwargs)

    def identifier_doi(self, **kwargs) -> str:
        """
        Generator for doi identifiers
        Parameters
        ----------

        Returns
        -------
            doi identifier

        """
        yield from self._doc_2_str_gen_s(attribute='prism:doi', **kwargs)

    def identifier_issn(self, **kwargs) -> str:
        """
        Generator for issn identifiers
        Parameters
        ----------

        Returns
        -------
            issn identifier

        """
        yield from self._doc_2_str_gen_s(attribute='prism:issn', **kwargs)

    def identifier_pubmed(self, **kwargs) -> str:
        """
        Generator for issn identifiers
        Parameters
        ----------

        Returns
        -------
            issn identifier

        """
        yield from self._doc_2_str_gen_s(attribute='pubmed-id', **kwargs)

    def identifier_source(self, **kwargs) -> str:
        """
        Generator for issn identifiers
        Parameters
        ----------

        Returns
        -------
            issn identifier

        """
        yield from self._doc_2_str_gen_s(attribute='source-id', **kwargs)

    def publication_type(self, **kwargs) -> str:
        """
        Generator for publication type
        Parameters
        ----------

        Returns
        -------
            publication type

        """
        yield from self._doc_2_str_gen_s(attribute='prism:aggregationType',
                                         **kwargs)

    def publication_name(self, **kwargs) -> str:
        """
        Generator for publication name
        Parameters
        ----------

        Returns
        -------
            publication name

        """
        yield from self._doc_2_str_gen_s(attribute='prism:publicationName',
                                         **kwargs)

    def publication_subtype(self, **kwargs) -> str:
        """
        Generator for publication subtype
        Parameters
        ----------

        Returns
        -------
            publication subtype

        """
        yield from self._doc_2_str_gen_s(attribute='subtypeDescription',
                                         **kwargs)

    def publication_volume(self, **kwargs) -> str:
        """
        Generator for publication volume Identifier, Some values are not
        integers, so this method will return a string
        Parameters
        ----------

        Returns
        -------
            publication volume number

        """
        yield from self._doc_2_str_gen_s(attribute='prism:volume',
                                         **kwargs)

    def publication_issue(self, **kwargs) -> str:
        """
        Generator for publication issue number, some publications make use of
        month or some other string value so to be conservative I will make these
         all strings.
        Parameters
        ----------

        Returns
        -------
            publication issue number

        """
        yield from self._doc_2_str_gen_s(attribute='prism:issueIdentifier',
                                         **kwargs)

    def publication_pages(self, **kwargs) -> (int, int):
        """
        Generator for publication page numbers a tuple containing -1
        indicates missing data
        Parameters
        ----------

        Returns
        -------
            publication page numbers

        """
        for doc in self.docs(**kwargs):
            if doc:
                try:
                    yield tuple(int(p) for p in
                                doc['prism:pageRange'].split('-'))
                except (KeyError, AttributeError, ValueError):
                    yield (-1, -1)
            else:
                yield (-1, -1)

    def publication_date(self, **kwargs) -> object:
        """
        Generator for publication cover date, missing data is replaced by
        '1800-01-01'
        Parameters
        ----------

        Returns
        -------
            publication cover date

        """
        for doc in self.docs(**kwargs):
            if doc:
                try:
                    date_string = doc['prism:coverDate']
                    yield datetime.strptime(date_string, '%Y-%m-%d')
                except KeyError:
                    date_string = '1800-01-01'
                    yield datetime.strptime(date_string, '%Y-%m-%d')
            else:
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
        yield from self._doc_2_str_gen_s(attribute='dc:title',
                                         **kwargs)

    def document_description(self, **kwargs) -> str:
        """
        generates the description of the next document in the corpus
        Parameters
        ----------

        Returns
        -------
            yields a string containing the next document description

        """
        yield from self._doc_2_str_gen_s(attribute='dc:description',
                                         **kwargs)

    def describe(self, **kwargs) -> dict:

        started = time.time()

        # ensure fileids and catagories are passed to resolve.
        if not any(key.startswith('fileids') for key in kwargs.keys()):
            kwargs['fileids'] = None

        if not any(key.startswith('categories') for key in kwargs.keys()):
            kwargs['categories'] = None

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(**kwargs) or self.fileids())
        n_topics = len(self.categories(self.resolve(**kwargs)))

        # Structures to perform counting.
        # counts = nltk.FreqDist()
        counts = defaultdict(int)

        gen_map = {
            'affiliation_city_s': self.affiliation_city_s(**kwargs),
            'affiliation_country_s': self.affiliation_country_s(**kwargs),
            'affiliation_name_s': self.affiliation_name_s(**kwargs),
            'affiliation_id_s': self.affiliation_id_s(**kwargs),
            'keywords_phrase': self.keywords_phrase(**kwargs),
            'author_data_id_s': self.author_data_id_s(**kwargs),
            'author_data_name_full_s': self.author_data_name_full_s(**kwargs),
            'publication_type': self.publication_type(**kwargs),
            'publication_name': self.publication_name(**kwargs),
            'document_title': self.document_title(**kwargs),
            'document_description': self.document_description(**kwargs),
                   }
        # Perform single pass over each generator and count occurance
        for name, generator in gen_map.items():
            unique_values = set()
            for item in generator:
                if item is not 'unk':
                    # counts[name] += 1
                    unique_values.add(item)
                    counts['{}_unique'.format(name)] = len(unique_values)

        # Return data structure with information

        counts['files'] = n_fileids
        counts['topics'] = n_topics
        counts['secs'] = time.time() - started

        return counts


class ScopusProcessedCorpusReader(ScopusCorpusReader):
    """
    adds additional features to the ScopusCorpusReader that include text
    specific tools.
    """

    def __init__(self, root, **kwargs):
        """
        Initialise the ScopusCorpusReader. any additional arguements are
        passed directly to ScopusCorpusReader
        Parameters
        ----------
        root : str like
            the root directory for the corpus
        kwargs :
            Additional arguements passed to ScopusCorpusReader
        """
        ScopusCorpusReader.__init__(self, root, **kwargs)

    def title_tagged(self, **kwargs) -> list:
        """
        Yields the next title as a list of sentences that are a list of
        tagged words
        Parameters
        ----------

        Returns
        -------

        """
        for doc in self.docs(**kwargs):
            yield doc['processed:dc:title']

    def title_tagged_sents(self, **kwargs) -> list:
        """
        Gets the next title sentence
        Parameters
        ----------

        Returns
        -------
            yields a tagged list fo tuples

        """
        for title in self.title_tagged(**kwargs):
            for sent in title:
                yield sent

    def title_tagged_word(self, **kwargs) -> (str, str):
        """
        yields the next tagged word in the title
        Parameters
        ----------

        Returns
        -------
            yields the next tagged word in the title

        """
        for sent in self.title_tagged_sents(**kwargs):
            for tagged_token in sent:
                yield tagged_token

    def title_word(self, **kwargs) -> str:
        """
        yields the next word in the title
        Parameters
        ----------

        Returns
        -------
            yields the next word in the title
        """
        for word, token in self.title_tagged_word(**kwargs):
            yield word

    def title_ngrams(self, n=2, **kwargs) -> tuple:
        """
        a ngram generator for the scopus corpus
        Parameters
        ----------
        n : int
            the number of words in the n-gram

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
        for sent in self.title_sents(**kwargs):
            tokens, _ = zip(*sent)
            for ngram in nltk_ngrams(tokens, n):
                yield ngram

    def description_tagged(self, **kwargs) -> list:
        """
        Yields the next description as a list of sentences that are a list of
        tagged words
        Parameters
        ----------

        Returns
        -------

        """
        for doc in self.docs(**kwargs):
            yield doc['processed:dc:description']

    def description_tagged_sents(self, **kwargs) -> list:
        """
        Gets the next description sentence
        Parameters
        ----------

        Returns
        -------
            yields a tagged list fo tuples

        """
        for description in self.description_tagged(**kwargs):
            for sent in description:
                yield sent

    def description_tagged_word(self, **kwargs) -> (str, str):
        """
        yields the next tagged word in the description
        Parameters
        ----------

        Returns
        -------
            yields the next tagged word in the title

        """
        for sent in self.description_tagged_sents(**kwargs):
            for tagged_token in sent:
                yield tagged_token

    def description_word(self, **kwargs) -> str:
        """
        yields the next word in the description
        Parameters
        ----------

        Returns
        -------
            yields the next word in the title
        """
        for word, token in self.description_tagged_word(**kwargs):
            yield word

    def description_ngrams(self, n=2, **kwargs) -> tuple:
        """
        a ngram generator for the scopus corpus
        Parameters
        ----------
        n : int
            the number of words in the n-gram

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
        for sent in self.description_sents(**kwargs):
            tokens, _ = zip(*sent)
            for ngram in nltk_ngrams(tokens, n):
                yield ngram

    def describe(self, **kwargs) -> dict:

        started = time.time()

        # ensure fileids and catagories are passed to resolve.
        if not any(key.startswith('fileids') for key in kwargs.keys()):
            kwargs['fileids'] = None

        if not any(key.startswith('categories') for key in kwargs.keys()):
            kwargs['categories'] = None

        # Structures to perform counting.
        counts = nltk.FreqDist()
        t_tokens = nltk.FreqDist()

        # Perform single pass over paragraphs, tokenize and count
        for title in self.document_title(**kwargs):
            counts['titles'] += 1

            for word in wordpunct_tokenize(title):
                counts['t_words'] += 1
                t_tokens[word] += 1

        d_tokens = nltk.FreqDist()
        for description in self.document_description(**kwargs):
            counts['description'] += 1

            for word in wordpunct_tokenize(description):
                counts['d_words'] += 1
                d_tokens[word] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(**kwargs) or self.fileids())
        n_topics  = len(self.categories(self.resolve(**kwargs)))

        # Return data structure with information
        return {
            'files':  n_fileids,
            'topics': n_topics,
            'titles':  counts['titles'],
            't_words':  counts['t_words'],
            't_vocab':  len(t_tokens),
            't_lexdiv': float(counts['t_words']) / float(len(t_tokens)),
            't_tpdoc':  float(counts['titles']) / float(n_fileids),
            't_wptit':  float(counts['t_words']) / float(counts['titles']),
            'description': counts['description'],
            'd_words': counts['d_words'],
            'd_vocab': len(t_tokens),
            'd_lexdiv': float(counts['d_words']) / float(len(t_tokens)),
            'd_tpdoc': float(counts['description']) / float(n_fileids),
            'd_wptit': float(counts['d_words']) / float(counts['description']),
            'secs':   time.time() - started,
        }


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
    pp = PrettyPrinter(indent=4)

    # --------------------------------------------------------------------------
    # RawCorpusReader
    # --------------------------------------------------------------------------
    # root = "Tests/Test_Corpus/Raw_corpus/"
    # corpus = RawCorpusReader(root=root)
    # pp.pprint(corpus.describe())

    # --------------------------------------------------------------------------
    # ScopusCorpusReader
    # --------------------------------------------------------------------------
    # root = "Tests/Test_Corpus/Split_corpus/"
    # corpus = ScopusCorpusReader(root=root)
    # pp.pprint(corpus.describe())

    # --------------------------------------------------------------------------
    # ScopusProcessedCorpusReader
    # --------------------------------------------------------------------------
    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = ScopusProcessedCorpusReader(root=root)
    pp.pprint(corpus.describe())


