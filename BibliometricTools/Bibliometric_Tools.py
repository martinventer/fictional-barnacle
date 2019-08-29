#! envs/fictional-barnacle/bin/python3.6
"""
Elsevier_Corpus_Reader.py

@author: martinventer
@date: 2019-08-28

Tools for processing bibliometric data contained in
"""
from CorpusReaders import Elsevier_Corpus_Reader
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns


class CorpusFrameGenerator:
    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_frame = None

    def generate_frame(self) -> pd.DataFrame:
        self.corpus_frame = pd.DataFrame(list(self.corpus.docs()))
        self.corpus_frame.drop(columns=["@_fa"], inplace=True)
        self.corpus_frame.set_index(
            self.corpus_frame["file_name"], inplace=True)
        return self.corpus_frame


def extract_data_from_dict_cell(dict_cell, dict_key) -> str:
    """
    helper function that can replace a cell containing a dictionary with an
    item from within that dictionary
    Parameters
    ----------
    dict_cell : dict
        a dictionary
    dict_key : str
        a dictionary key
    Returns
    -------
        a single dict item as a string
    """
    if type(dict_cell) is dict:
        return str(int(dict_cell[dict_key]))
    else:
        return dict_cell

class AuthorInfo:
    """
    wrapper for corpus for presenting author metrics
    """
    def __init__(self, corpus):
        """
        Initialise a corpus frame
        Parameters
        ----------
        corpus
        """
        corpus_frame_generator = CorpusFrameGenerator(corpus)
        self.corpus_frame = corpus_frame_generator.generate_frame()

    def papers_per_author(self):
        column_of_interest = 'author'
        dict_key_of_interest = 'authid'

        # extract a list cell from the the data fram and create a new fram
        # with the list items in their own columns
        a = corpus_frame.loc[:, column_of_interest].apply(pd.Series)

        # extract just the desired item from the separated dictionaries
        b = a.applymap(
            lambda x: extract_data_from_dict_cell(
                x, dict_key=dict_key_of_interest)
        )
        b.reset_index(level=0, inplace=True)

        # reshape the frame into the long form for further processing
        c = pd.melt(
            b,
            id_vars=['file_name'],
            value_name=dict_key_of_interest)
        c.drop(['variable'], axis=1, inplace=True)
        # remove missing data
        c.dropna(subset=[dict_key_of_interest], inplace=True)

        d = c.groupby([dict_key_of_interest]).count()
        d.columns = ['count']


class PublicationInfo:
    """
    wrapper for corpus for presenting publication met metrics
    """
    def __init__(self, corpus):
        """
        Initialise a corpus frame
        Parameters
        ----------
        corpus
        """
        corpus_frame_generator = CorpusFrameGenerator(corpus)
        self.corpus_frame = corpus_frame_generator.generate_frame()

    def describe(self) -> None:
        """
        Provides a summary of the basic data regarding a database publication
        metadata including
            # >>Total number of publications (sum of all unique datapoints)
            # >>Total number of citations (sum of citation count)
            # >>Citations per publication item
            # >>Distribution top 20 publication subtypes (subtypeDescription
            , book or  artical ...?. by count or by citation)
            # Distribution top 20 publication names (prism:publicationName. by count or by citation)
            # Distribution top 20 aggregation types (prism:aggregationType. by count or by citation)
            # publication count by first author country
            # publication count by first author affiliation
            # map of paper count by first author country or affiliation
        Returns
        -------
            None
        """
        self._totals()
        self._publication_distribution('subtypeDescription')
        # self._publication_distribution('prism:publicationName')
        # self._distribution_publication()


    def _totals(self) -> None:
        """
        Prints the total number of publications, total number of citations
        and the average number of citations per publication
        Returns
        -------
            None
        """
        total_publications = self.corpus_frame.shape[0]
        print("Total Number of Published Items: {}".format(
            total_publications))

        total_citations = self.corpus_frame.loc[:, 'citedby-count']\
            .dropna()\
            .astype(int)\
            .sum()
        print("Total Number of Citations: {}".format(
            total_citations))

        citations_per_paper = total_citations/total_publications
        print("Average number of Citations per Publication Item: {}".format(
            citations_per_paper))

    def _publication_distribution(self, field) -> None:
        """
        Plots the number of publications per search field
        Returns
        -------
            None
        """
        self.corpus_frame.groupby(
            [field],
        )[field].count(
        ).sort_values(ascending=False).plot.bar()

    def _distribution_publication(self) -> None:
        """
        Plots the distribution of publications by publication subtype
        Returns
        -------
            None
        """
        self.corpus_frame.groupby(
            ['prism:publicationName'],
        )['prism:publicationName'].count(
        ).sort_values(ascending=False).plot.bar()





if __name__ == '__main__':
    # ==========================================================================
    # Authornfor
    # ==========================================================================
    if False:
        from collections import Counter
        import seaborn as sns
        root = "Corpus/Split_corpus/"
        corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            root=root)
        # corpus_frame_generator = CorpusFrameGenerator(corpus)
        # corpus_frame = corpus_frame_generator.generate_frame()

        column_of_interest = 'author'
        dict_key_of_interest = 'authid'

        # extract a list cell from the the data fram and create a new fram
        # with the list items in their own columns
        a = corpus_frame.loc[:, column_of_interest].apply(pd.Series)

        # extract just the desired item from the separated dictionaries
        b = a.applymap(
            lambda x: extract_data_from_dict_cell(
                x, dict_key=dict_key_of_interest)
        )
        b.reset_index(level=0, inplace=True)

        # reshape the frame into the long form for further processing
        c = pd.melt(
            b,
            id_vars=['file_name'],
            value_name=dict_key_of_interest)
        c.drop(['variable'], axis=1, inplace=True)
        # remove missing data
        c.dropna(subset=[dict_key_of_interest], inplace=True)

        d = c.groupby([dict_key_of_interest]).count()
        d.columns = ['count']
        # f = Counter(d['author_id'])
        # f.most_common(5)
        #
        # g = d.groupby(['file_name']).count()
        # g.columns = ['paper_count']
        # h = Counter(d['file_name'])
        # h.most_common(5)
        #
        # e.hist(bins=234)
        #
        #
    # ==========================================================================
    # PublicationInfo
    # ==========================================================================
    if True:


        root = "Corpus/Split_corpus/"
        corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            root=root)
        pub_meta = PublicationInfo(corpus=corpus)
        pub_meta.describe()
