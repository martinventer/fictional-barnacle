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


class CorpusFrameGenerator:
    def __init__(self, root):
        self.root = root
        self.corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            root=root)
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


if __name__ == '__main__':
    # ==========================================================================
    # CorpusReaders
    # ==========================================================================
    if True:
        from collections import Counter
        import seaborn as sns
        root = "Corpus/Split_corpus/"
        corpus_frame_generator = CorpusFrameGenerator(root=root)
        corpus_frame = corpus_frame_generator.generate_frame()

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
