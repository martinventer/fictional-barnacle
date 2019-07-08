#! envs/fictional-barnacle/bin/python3.6
"""
Text_Visualization.py

@author: martinventer
@date: 2019-07-08

Tools for filtering and grouping a corpus.
"""

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


class Corpus2Frame():
    """
    transform corpus docs to df
    """
    def __init__(self):
        pass

    def fit(self, corpus=None):
        pass

    def transform(self, corpus, fileids=None, catagories=None):
        return pd.DataFrame(corpus.docs(fileids=fileids,
                                        categories=catagories))


if __name__ == '__main__':
    from CorpusReader import Elsevier_Corpus_Reader
    from CorpusProcessingTools import Corpus_Vectorizer

    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 12, shuffle=False)
    subset = next(loader.fileids(test=True))

    c_framer = Corpus2Frame()
    df = c_framer.transform(corpus, fileids=subset)