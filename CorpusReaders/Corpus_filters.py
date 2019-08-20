#! envs/fictional-barnacle/bin/python3.6
"""Corpus_filters.py

@author: martinventer
@date: 2019-07-17

Tools for filtering and grouping a corpus.These filters perform a single run
through the data update the subset that meets the focus
"""
from sklearn.model_selection import KFold


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

    def reset_subset(self, **kwargs):
        self.subset = corpus.fileids(**kwargs)

    def filter_by(self):

        include_subset, exclude_subset = self._by_publication()

        print("initial {} included {} excluded {}".format(
            len(self.subset), len(include_subset), len(exclude_subset)))

        if len(include_subset) is 0 and len(exclude_subset) is not 0:
            new_subset = list(set(self.subset) - set(exclude_subset))
            self.subset = new_subset

        elif len(exclude_subset) is 0 and len(include_subset) is not 0:
            self.subset = include_subset

        else:
            new_subset = list(set(include_subset) - set(exclude_subset))
            self.subset = new_subset

        return self.subset

    def _by_publication(self,
                        include_list=None,
                        exclude_list=None,
                        method="year"):
        self.filters.append({
            'publication': {
                "method": method,
                "include": include_list,
                "exclude": exclude_list}})



        return include_subset, exclude_subset


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

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)

    all_files = corpus.fileids()

    # ==========================================================================
    # FilteredCorpus
    # ==========================================================================
    if True:
        filtered_corpus = FilteredCorpus(corpus)
        subset = filtered_corpus.filter_by()

