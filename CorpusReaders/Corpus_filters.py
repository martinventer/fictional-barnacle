#! envs/fictional-barnacle/bin/python3.6
"""Corpus_filters.py

@author: martinventer
@date: 2019-07-17

Tools for filtering and grouping a corpus.These filters perform a single run
through the data update the subset that meets the focus
"""
from sklearn.model_selection import KFold

from CorpusReaders import Elsevier_Corpus_Reader


class FilteredCorpus(object):
    """
    a wraper for the ScopusProcessedCorpusReader has its own fileids subset.
    The object also keeps track of what filtered have been applied and in
    what order.
    """
    def __init__(self, corpus):
        """
        Wrap an existing corpus reader
        Parameters
        ----------
        corpus
            ScopusProcessedCorpusReader
        """
        self._corpus = corpus
        self.subset = corpus.fileids()
        self.filters = []

    def affiliation_city(self,
                         include_list=None,
                         exclude_list=None) -> list:
        """
        Filter corpus by affiliation city, if both an include_list and
        exclude_list are provided the included list will be filtered by the
        exclude cases.
        Parameters
        ----------
        include_list : list
            list of terms that should be included in the dataset
        exclude_list : list
            list of terms to be excluded from the dataset

        Returns
        -------

        """
        # if there are no filters return the current subset.

        self.filters.append({'affiliation_city': {"include": include_list,
                                                  "exclude": exclude_list}})

        if not include_list and not exclude_list:
            return self.subset
        # else replace the either the include_list or exclude_list None value
        # with an empty list
        elif not include_list:
            include_list = []
        elif not exclude_list:
            exclude_list = []

        include_subset = []
        exclude_subset = []
        for fileid, city_list in zip(self.subset,
                                     self._corpus.affiliation_city_l(
                                         fileids=self.subset)):

            # check if an item in the include_list exists in the affiliations
            # of the current document
            if len(set(include_list) - set(city_list)) is not \
                    len(set(include_list)):
                include_subset.append(fileid)

            # check if an item in the exclude_list exists in the affiliations
            # of the current document
            if len(set(exclude_list) - set(city_list)) is not \
                    len(set(exclude_list)):
                exclude_subset.append(fileid)
                # print(fileid, city_list)

        print(len(self.subset), len(include_subset), len(exclude_subset))
        if len(include_subset) is 0 and len(exclude_subset) is not 0:
            new_subset = list(set(self.subset) - set(exclude_subset))
            self.subset = new_subset
            print(0)

        elif len(exclude_subset) is 0 and len(include_subset) is not 0:
            self.subset = include_subset
            print(1)

        else:
            new_subset = list(set(include_subset) - set(exclude_subset))
            self.subset = new_subset
            print(2)

        return self.subset


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)

    filtered_corpus = FilteredCorpus(corpus)
    all_files = corpus.fileids()

    # --------------------------------------------------------------------------
    # affiliation_city
    # --------------------------------------------------------------------------
    gen = corpus.affiliation_city_s()
    in_list = [next(gen) for i in range(10)]
    # out_list = [next(gen) for i in range(10)]
    out_list = ['unk']

    subset0 = filtered_corpus.affiliation_city()
    filtered_corpus = FilteredCorpus(corpus)
    subset1 = filtered_corpus.affiliation_city(include_list=in_list)
    filtered_corpus = FilteredCorpus(corpus)
    subset2 = filtered_corpus.affiliation_city(exclude_list=in_list)
    filtered_corpus = FilteredCorpus(corpus)
    subset3 = filtered_corpus.affiliation_city(include_list=in_list,
                                               exclude_list=out_list)


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