#! envs/fictional-barnacle/bin/python3.6
"""
Elsevier_Corpus_Reader.py

@author: martinventer
@date: 2019-06-10

Reads the raw data from Elsivier Ingestor and refactors it into a per article
"""

import pickle
import os
import hashlib
import time
import logging

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

from tqdm import tqdm

from Utils import Utils

logging.basicConfig(filename='logs/PreProcess.log',
                    format='%(asctime)s %(message)s',
                    level=logging.INFO)


def split_corpus(corpus, target) -> None:
    """
    Reads the raw corpus and generates a new file for each search entity.
    The new pickle files are stored in a single folder for each category
    in the corpus. That is per search term.
    Parameters
    ----------
    target :
        path to target directory
    corpus :
        corpus to be processed

    Returns
    -------

    """
    logging.info('Start')
    start = time.time()

    # create the target folder
    Utils.make_folder(target)

    # iterate through each file listed in the corpus fileids, split the
    # file name into term and year. The term is then used to create a
    # folder for each search term.
    for filename in tqdm(corpus.fileids(),
                         desc='refactor corpus'):
        term, year = filename.rstrip(".pickle").split("/")
        file_path = "{}/".format(term)
        file_path = os.path.join(target, file_path)
        Utils.make_folder(file_path)
        data = corpus.read_single(filename)
        for paper in data:
            hashstring = hashlib.md5(str(paper).encode('utf-8')).hexdigest()
            data_path = os.path.join(file_path,
                                     str(hashstring) + '.pickle')
            if len(paper) is not 0:
                with open(data_path, 'wb') as f:
                    pickle.dump(paper, f, pickle.HIGHEST_PROTOCOL)

    logging.info('End')
    end = time.time()
    print("Time to reformat corpus {}seconds".format(end - start))


class PickledCorpusPreProcessor(object):
    """
    A wrapper for a corpus object that reads the raw imported data
    and reformat sections to have a more suitable for text processing.
    """
    #also add the file name to the data

    def __init__(self, corpus, target=None):
        """
        The corpus is the `HTMLCorpusReader` to preprocess and pickle.
        The target is the directory on disk to output the pickled corpus to.
        """
        self.corpus = corpus
        self.target = target

    def fileids(self, fileids=None, categories=None):
        """
        Helper function access the fileids of the corpus
        """
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def tokenize(self, document):
        """
        Segments, tokenizes, and tags a document title in the corpus. Returns a
        title, which is a list of sentences, which in turn is a lists of part
        of speech tagged words.
        """
        try:
            return [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(document['dc:title'])
            ]
        except KeyError:
            pass

    def process(self, fileid):
        """
                For a single file does the following preprocessing work:
            1. Get the location of the document
            2. Generate a structured text list,
            3. Append the new structured text to the existing document
            4. Writes the document as a pickle to the target location.
            5. Clean up the document
            6. Return the target file name
        This method is called multiple times from the transform runner.
        Parameters
        ----------
        fileid : str
            the file id to be processed

        Returns
        -------
            None

        """
        # 1. Get the location of the document
        target = self.corpus.abspath(fileid)

        # 2 & 3. Generate and append structured form of text to document
        document = self.corpus.read_single(fileid)
        title_token = self.tokenize(document)
        try:
            if type(title_token[0]) is tuple:
                document["struct:title"] = [title_token]
            else:
                document["struct:title"] = title_token
        except TypeError:
            # pass
            document["struct:title"] = [[('', '')]]

        # 4. Writes the document as a pickle to the target location.
        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

        # 5. Clean up the document
        del document

        # 6. Return the target file name
        # return target

    def transform(self, fileids=None, categories=None):
        """
        Take an existing corpus and transform it such that it contains more
        suitably formatted data.

        process each file in the corpus
        Parameters
        ----------
        fileids: basestring or None
            complete path to specified file
        categories: basestring or None
            path to directory containing a subset of the fileids

        Returns
        -------
            None

        """
        start = time.time()
        # serial file processing
        for filename in tqdm(self.fileids(fileids, categories),
                         desc="transforming pickled corpus"):
            self.process(filename)

        # parallel file processing
        # with Pool(6) as p:
        #     tqdm(p.imap(self.process, self.fileids(fileids, categories)),
        #          desc="preprocess corpus parallel")

        end = time.time()
        print("Time to pre process {}seconds".format(end - start))
