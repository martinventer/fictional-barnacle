# /home/martin/Documents/RESEARCH/fictional-barnacle/Ingestor/
"""
Elsevier_Ingestor.py

@author: martinventer
@date: 2019-06-10

An Ingestion engine used to retrieve data form the Elsivier API
"""

import requests
from datetime import datetime
import os
import pickle
from tqdm import trange
from urllib.request import urlopen
import xmltodict
import logging


logging.basicConfig(filename='logs/Ingestor.log',
                    format='%(asctime)s %(message)s',
                    level=logging.INFO)


def get_classifications() -> list:
    """
    get a list of subject classifications associated with ScienceDirect content.
    Returns
    -------
        a list of subject classifications
    """
    file = urlopen(
        'https://api.elsevier.com/content/subject/scopus?httpAccept=text/xml')
    data = file.read()
    file.close()

    data = xmltodict.parse(data)['subject-classifications']
    data = data['subject-classification']
    return list(set([x['@abbrev']
                     for x
                     in data]))


def get_key(file='api_key.dict', key='ElsevierDeveloper'):
    """
    opens a file containing api keys and returns the specific key requested
    Parameters
    ----------
    file : str
        the name of the file containing the api keys
    key : str
        which api key is being requested

    Returns
    -------
    str
        api key as string
    """
    return eval(open(file, 'r').read())[key]


def make_folder(path):
    """
    creates a directory without failing unexpectedly
    Parameters
    ----------
    path : str
        a string containing the desired path

    Returns
    -------

    """
    try:
        os.makedirs(path)
    except FileExistsError:
        logging.error("file %s already exists" % path)
        pass


class ElsevierIngestionEngine(object):
    """
    An interface for the Elsevier search API capable of searching both the
    scopus and science direct databases, if you have the correct API key
    """

    def __init__(self,
                 search_terms=(),
                 dates=(1900, datetime.today().year),
                 home=False,
                 batch_size=25,
                 database='scopus',
                 file_path="corpus/"):
        """
        Initialize the elsevier searcher
        Parameters
        ----------
        database : str
            selection of which database to search
                'scopus' is default
                'science_direct'
        dates : object (list like)
            a list of dates covering the range of the search
        home : bool
            in order to get full access to the databases you will need to be
            on a network that has access
        batch_size : int
            the number of entities that will be downloaded in a single batch
        file_path : str
            the path to where the corpus should be stored
        """
        self.file_path = os.path.dirname(file_path)
        self.batch_size = batch_size
        self.home = home
        self.search_terms = search_terms
        self.dates = dates
        self.dates_range = range(*dates)
        self.database = database
        self.api_key = get_key()
        self.status_code = None
        logging.info("Corpus Builder Initialised")

    def search_by_term(self,
                       search_term,
                       start=0,
                       date=datetime.today().year,
                       subject=None):
        """
        a basic search of the science direct database
        Parameters
        ----------
        subject : int
            subject numeric code
        search_term : str
            a string representing the term of interest to be searched for.
        start : int
            an index offset used when doing a series of consecutive searches
        date : int
            date filter for search, default is current year

        Returns
        -------
        json
            a dictionary like object containing the search results
        """
        headers = {"X-ELS-APIKey": self.api_key}
        view = 'COMPLETE' if (self.home is False) else 'STANDARD'
        endpoints = {'scopus' :
                         "https://api.elsevier.com/content/search/scopus",
                     'science_direct':
                         "https://api.elsevier.com/content/search/sciencedirect"}

        endpoint = endpoints[self.database]

        if subject is None:
            params = {"query": search_term,
                      "start": start,
                      "count": self.batch_size,
                      "date": date,
                      "view": view}
        else:
            params = {"query": search_term,
                      "start": start,
                      "count": self.batch_size,
                      "date": date,
                      "subj": subject,
                      "view": view}

        search_results = requests.get(endpoint,
                                      headers=headers,
                                      params=params)

        self.status_code = search_results.status_code
        try:
            return search_results.json()['search-results']
        except KeyError:
            logging.error(
                'connection fails with status %d' % search_results.status_code)
            logging.error(
                '\t search term %s starting at %d' % (search_term, start))
            return None

    def get_num_batches(self, num_entities) -> int:
        """
        Looks at the total number of entities found and calculates the number
        of batches
        Parameters
        ----------
        num_entities: int
            THe number of entities returned in a given search.

        Returns
        -------

        """
        if (num_entities % self.batch_size) is not 0:
            num_batches = int(num_entities / self.batch_size) + 1
        else:
            num_batches = int(num_entities / self.batch_size)
        return num_batches

    def retrieve_all_in_year(self, term, year):
        """
        retrieves all entities associated with a give search term in a given
        year
        Parameters
        ----------
        term : str
            string containing the search term of interest
        year : int
            the year of interest for the search

        Returns
        -------
        list
            returns a list of related to the search term in the given year
        """
        results_year = list()
        batch_start = 0

        search_results = self.search_by_term(term, start=batch_start, date=year)
        expected_num_of_ent = int(search_results["opensearch:totalResults"])
        if self.status_code is not 200 or expected_num_of_ent is 0:
            logging.info(" %s in year %d contains no results" % (term, year))
            pass

        if 0 < expected_num_of_ent < 5000:
            num_batches = self.get_num_batches(expected_num_of_ent)
            for batch in trange(num_batches, ascii=True, desc=str(year)):
                batch_start = self.batch_size * batch
                try:
                    search_results = self.search_by_term(term,
                                                         start=batch_start,
                                                         date=year)
                    for entry in search_results['entry']:
                        results_year.append(entry)
                except EOFError:
                    logging.error(
                        "failed to retrieve %s in year %d" % (term, year))
                    break
        elif expected_num_of_ent >= 5000:
            logging.error(
                "more than 5000 entries expected for  %s in year %d" % (
                term, year))
            list_of_subjects = get_classifications()
            for subject in list_of_subjects:
                batch_start = 0
                search_results = self.search_by_term(term, start=batch_start,
                                                     date=year, subject=subject)
                expected_num_of_ent = int(
                    search_results["opensearch:totalResults"])
                if self.status_code is not 200 or expected_num_of_ent is 0:
                    logging.info(
                        " %s in year %d contains no results" % (term, year))
                    pass

                num_batches = self.get_num_batches(expected_num_of_ent)
                for batch in trange(num_batches, ascii=True,
                                    desc=str(year)+str(subject)):
                    batch_start = self.batch_size * batch
                    search_results = self.search_by_term(term,
                                                         start=batch_start,
                                                         date=year,
                                                         subject=subject)
                    try:
                        for entry in search_results['entry']:
                            results_year.append(entry)
                    except:
                        logging.error(
                            "failed to retrieve %s in year %d" % (term, year))
                        break

        return results_year

    def build_corpus(self):
        """
        takes the corpus builder initialised with the search terms and dates then iterates over each term for each
        year, saving the data in files by each year in a folder for each term.
        Returns
        -------
        NONE
            builds a pickled database of the data returned.
        """
        logging.info('Start')

        make_folder(self.file_path)
        self.gen_info_file()

        for term in self.search_terms:
            term_path = os.path.join(self.file_path, term)
            make_folder(term_path)
            logging.info("searching for %s" % term)

            for year in self.dates_range:
                logging.error(
                    "Start retrieving %s in year %d" % (term, year))
                data_path = os.path.join(term_path, str(year) + '.pickle')
                data = self.retrieve_all_in_year(term, year)
                if len(data) is not 0:
                    with open(data_path, 'wb') as f:
                        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        logging.info('End')

    def gen_info_file(self):
        readme_location = self.file_path + '/info.txt'
        with open(readme_location, "w") as f:
            f.write("source database : %s \n" % self.database)
            f.write("search between %d and %d \n" % self.dates)
            for term in self.search_terms:
                f.write("-- %s \n" % term)
