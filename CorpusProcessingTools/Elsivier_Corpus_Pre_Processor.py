# /home/martin/Documents/RESEARCH/fictional-barnacle/CorpusProcessingTools/
"""
Elsevier_Corpus_Reader.py

@author: martinventer
@date: 2019-06-10

Reads the raw data from Elsivier Ingestor and refactors it into a per article
"""

import pickle
import os
import hashlib

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

from tqdm import  tqdm

import logging


logging.basicConfig(filename='logs/PreProcess.log',
                    format='%(asctime)s %(message)s',
                    level=logging.INFO)


PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([0-9_\s]+)/.*'


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


class PickledCorpusPreProcesor(CategorizedCorpusReader, CorpusReader):

    def __init__(self, root, target, fileids=PKL_PATTERN, **kwargs):
        """
        Initialise the pickled corpus Pre_processor using two corpus readers
        from the nltk library
        Parameters
        ----------
        target : str like
            the target directory for the corpus
        root : str like
            the root directory for the corpus
        fileids : str like
            a regex pattern for the corpus document files
        kwargs :
            Additional arguements passed to the nltk corpus readers
        """
        # Add the default category pattern if not passed into the class.
        self.target = os.path.dirname(target)
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids, categories):
        """
         Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. This primarily bubbles up to
        the high level ``docs`` method, but is implemented here similar to
        the nltk ``CategorizedPlaintextCorpusReader``.
        Parameters
        ----------
        fileids :
        categories :

        Returns
        -------

        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. Similar to the BaleenCorpusReader, this uses a generator
        to archeive memory safe iteration.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def read_single(self, fileid=None, root=None):
        root = self.root if (root is None) else root
        # print(root)
        with open(os.path.join(root, fileid), 'rb') as f:
            return pickle.load(f)

    def refactor_corpus(self):
        """
        takes the corpus builder initialised with the search terms and dates then iterates over each term for each
        year, saving the data in files by each year in a folder for each term.
        Returns
        -------
        NONE
            builds a pickled database of the data returned.
        """
        logging.info('Start')

        make_folder(self.target)

        # for term in self.categories():
        #     term_path = os.path.join(self.target, term)
        #     make_folder(term_path)

        for filename in tqdm(self.fileids()):
            term, year = filename.rstrip(".pickle").split("/")
            # file_path = "{}_{}".format(term, year)
            file_path = "{}/{}/".format(term, year)
            # file_path= filename.rstrip(".pickle")#.split("/")
            file_path = os.path.join(self.target, file_path)
            make_folder(file_path)
            data = self.read_single(filename)
            for paper in data:
                hashstring = hashlib.md5(str(paper).encode('utf-8')).hexdigest()
                # paper_name = "{0}.pickle".format(hashstring)
                data_path = os.path.join(file_path,
                                         str(hashstring) + '.pickle')
                if len(paper) is not 0:
                    with open(data_path, 'wb') as f:
                        pickle.dump(paper, f, pickle.HIGHEST_PROTOCOL)

        logging.info('End')
