#! envs/fictional-barnacle/bin/python3.6
"""Corpus_filters.py

@author: martinventer
@date: 2019-07-17

Tools for filtering and grouping a corpus.These filters perform a single run
through the data update the subset that meets the focus
"""
from sklearn.model_selection import KFold
import copy


class FilteredCorpus(object):
    """
    A warpper for a corpus object that can return a subset view of the corpus.
    """
    def __init__(self, corpus):
        """
        Parameters
        ----------
        corpus
            ScopusProcessedCorpusReader
        """
        self._corpus = corpus
        self._view = copy.copy(self._corpus)
        self.subset_ids = corpus.fileids()
        self.filters = []

    def reset_subset(self):
        """
        reset the corpus view, subset_ids and filters
        """
        self._view = copy.copy(self._corpus)
        self.subset_ids = self._corpus.fileids()
        self.filters = []

    def view(self) -> object:
        """
        returns the filtered subset object
        Returns
        -------
            current filtered corpus
        """
        return self._view

    def filter(self,
               how=None,
               include_list=None
               ) -> list:
        """
        applies a simple filter to the corpus. A generator object for one
        corpus component is created. the method iterates over each component
        and compairs the resultd to the desired list. In this process a list
        of fileids meeting the filter criteria is created. Once complete the
        method updates the view such that it only contains the subset fileids.
        Parameters
        ----------
        how : str
            a string pertaining to which generator should be used
            for example :
                'publication_year'
                'publication_type'
        include_list : list
            a list of desired atributes that will be kept during filtering
        Returns
        -------
            a list of fileids meeting the filter criteria
        """

        # add current filter to the list of filters applied
        self.filters.append(
            {"method": how,
             "include": include_list})

        # create the filter method
        filter_field = getattr(self._view, how)

        # create a subset list that applies the filter to the current corpus
        # view
        self.subset_ids = [
            filen for filen, year
            in zip(self._view._fileids, filter_field())
            if year in include_list]

        # update the current view of the corpus
        self._view._fileids = self.subset_ids

        return self.subset_ids


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
    from CorpusReaders import Elsevier_Corpus_Reader
    from pprint import PrettyPrinter

    pp = PrettyPrinter(indent=4)

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)

    all_files = corpus.fileids()

    # ==========================================================================
    # FilteredCorpus
    # ==========================================================================
    if True:
        filtered_corpus = FilteredCorpus(corpus)
        # # pp.pprint(filtered_corpus._corpus.describe())
        # # pp.pprint(filtered_corpus.view().describe())

        list_of_dates = [1800, 2015]
        temp = filtered_corpus.filter(
            how='publication_year',
            include_list=list_of_dates)
        print(len(temp))

        list_of_dates = ['Conference Paper', "Article"]
        temp = filtered_corpus.filter(
            how='publication_subtype',
            include_list=list_of_dates)
        print(len(temp))

        list_of_dates = ['Conference Paper', "Article"]
        filtered_corpus.reset_subset()
        temp = filtered_corpus.filter(
            how='publication_subtype',
            include_list=list_of_dates)
        print(len(temp))


        #
        gen = filtered_corpus.view().publication_subtype()
        print(set(gen))
