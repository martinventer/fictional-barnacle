# /home/martin/Documents/RESEARCH/fictional-barnacle/Ingestor/
"""
Elsivier_Ingestor.py

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

from Utils import Utils

try:
    logging.basicConfig(filename='logs/Ingestor.log',
                        format='%(asctime)s %(message)s',
                        level=logging.INFO)
except:
    pass


def get_classifications() -> list:
    """
    get a list of subject classifications associated with ScienceDirect content.
    Returns
    -------
        A list of subject classifications for a given search term.
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


class ScopusIngestionEngine(object):
    """
    An interface for the Elsevier search API capable of searching the scopus
    databases. This can take in a list of search topics and a range of dated
    to be searched. A single pickle file is generated for all search results
    returned within a given year.
    """

    def __init__(self,
                 home=False,
                 batch_size=25,
                 file_path="Corpus/raw_corpus"):
        """
        Initialize the Scopus ingestion engine
        Parameters
        ----------
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
        self.api_key = Utils.get_key()
        self.status_code = None
        self.database = 'SCOPUS'
        self.endpoint = "https://api.elsevier.com/content/search/scopus"
        logging.info("Corpus Builder Initialised")

    def search_by_term(self,
                       search_term,
                       start=0,
                       date=datetime.today().year,
                       subject=None) -> dict:
        """
        Single Scopus search, for one term in one year.
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
            a dictionary like object containing the search results
        """
        headers = {"X-ELS-APIKey": self.api_key}
        view = 'COMPLETE' if (self.home is False) else 'STANDARD'

        params = {"query": search_term,
                  "start": start,
                  "count": self.batch_size,
                  "date": date,
                  "view": view}

        if subject is not None:
            params["subj"] = subject

        search_results = requests.get(self.endpoint,
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
            return {}

    def get_num_batches(self, num_entities) -> int:
        """
        Looks at the total number of entities found and calculates the number
        of download batches required, based on the batch size. The API is
        restricted to batches of 25 entities.
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

    def retrieve_all_in_year(self, term, year) -> list:
        """
        Retrieves all entities associated with a single search term in a given
        year.
        Parameters
        ----------
        term : str
            string containing the search term of interest
        year : int
            the year of interest for the search

        Returns
        -------
        list
            returns a list containg the data retrieved from the API
        """
        results_year = list()
        batch_start = 0

        # test the connection to the servers and determine the number of
        # entities to expect for the given search parameters.
        search_results = self.search_by_term(term, start=batch_start, date=year)
        expected_num_of_ent = int(search_results["opensearch:totalResults"])
        if self.status_code is not 200 or expected_num_of_ent is 0:
            logging.info(" %s in year %d contains no results" % (term, year))
            pass

        # If there are entities in the search. Download the entities in
        # batches. The API limits the number of entities that can be
        # downloaded through the API to 5000. If the number of entities is less
        # than 5000, a general search will provide all the entities.
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
                except KeyError:
                    logging.warning(
                        "no data to retrieve %s in year %d" % (term, year))
                    break
        # if the number of entities exceeds 5000 the search must be further
        # refined to searches by subject field, such as medical or
        # engineering. This extended search iterates over each subject field
        # within the general search parameters to work around the 5000 word
        # limit.
        elif expected_num_of_ent >= 5000:
            logging.warning(
                "more than 5000 entries expected for  %s in year %d"
                % (term, year))
            # get a list of classifications for the search parameters.
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
                    except KeyError:
                        logging.warning(
                            "no data to retrieve %s in year %d"
                            % (term, year))
                        break

        return results_year

    def build_corpus(self,
                     search_terms=None,
                     dates=(1900, datetime.today().year)) -> str:
        """
        Build and save a raw corpus for the given search terms, over the
        desired range of dates. Results are stored in pickle files in the
        provided directory path.
        Parameters
        ----------
        dates : tuple
            (start year, end year)
        search_terms : list
            list of search term strings

        Returns
        -------
            'FAIL' if input is incorrect
            'PASS' if search completes

        """
        logging.info('Start building corpus')

        if search_terms is None:
            search_terms = []

        if len(search_terms) is 0:
            logging.error("No search_terms provided")
            return 'FAIL'

        if len(dates) is not 2:
            logging.error("Date tuple provided is incorrect")
            return 'FAIL'

        Utils.make_folder(self.file_path)
        self.gen_info_file(terms=search_terms,
                           dates=dates)

        dates_range = range(*dates)

        for term in search_terms:
            term_path = os.path.join(self.file_path, term)
            Utils.make_folder(term_path)
            logging.info("searching for %s" % term)

            for year in dates_range:
                logging.error(
                    "Start retrieving %s in year %d" % (term, year))
                data_path = os.path.join(term_path, str(year) + '.pickle')
                data = self.retrieve_all_in_year(term, year)
                if len(data) is not 0:
                    with open(data_path, 'wb') as f:
                        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        logging.info('End')
        return 'PASS'

    def gen_info_file(self, terms, dates) -> None:
        """
        generates a text file containing the details of the most recient search
        Parameters
        ----------
        terms : list
            list of search terms provided
        dates : tuple
            start and end years for search (start, end)

        Returns
        -------

        """
        readme_location = self.file_path + '/info.txt'
        with open(readme_location, "w") as f:
            f.write("Search Time stamp {}".format(datetime.today()))
            f.write("Source database : %s \n" % self.database)
            f.write("Search from home {}".format(self.home))
            f.write("search between %d and %d \n" % dates)
            for term in terms:
                f.write("-- %s \n" % term)


class SciDirIngestionEngine(ScopusIngestionEngine):
    """
    An interface for the Elsevier search API capable of searching both the
    scopus and science direct databases, if you have the correct API key
    """

    def __init__(self,
                 home=False,
                 batch_size=25,
                 file_path="Corpus/"):
        """
        Initialize the elsevier searcher
        Parameters
        ----------
        home : bool
            in order to get full access to the databases you will need to be
            on a network that has access
        batch_size : int
            the number of entities that will be downloaded in a single batch
        file_path : str
            the path to where the corpus should be stored
        """
        ScopusIngestionEngine.__init__(self,
                                       home=home,
                                       batch_size=batch_size,
                                       file_path=file_path)
        self.database = 'Science Direct'
        self.endpoint = "https://api.elsevier.com/content/search/sciencedirect"
        logging.info("Corpus Builder Initialised")
