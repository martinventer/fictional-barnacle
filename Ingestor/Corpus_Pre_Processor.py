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

try:
    logging.basicConfig(filename='logs/PreProcess.log',
                        format='%(asctime)s %(message)s',
                        level=logging.INFO)
except FileNotFoundError:
    pass


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
    logging.info('Start splitting raw corpus files')
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


class ScopusCorpusProcessor(object):
    """
    wrapper for a SplitCorpusReader that adds processed data to the file.
    Data processing includes:
        * tokenize text into format where one document returns
            [list of paragraphs
                [list of sentences
                    [list of tagged tokens
                        (token, tag)]]]
        * tokenize additional text fields such as author, city, journal names
        * add the file path each document
    """

    def __init__(self, corpus, target=None):
        """
        The corpus is the `HTMLCorpusReader` to preprocess and pickle.
        The target is the directory on disk to output the pickled corpus to.

        Parameters
        ----------
        target : str
            path to store new files
        corpus : object
            a SplitCorpusReader object
        """
        self.corpus = corpus
        if target is not None:
            self.target = target
        else:
            self.target = self.corpus.root

    def fileids(self, **kwargs):
        """
        Helper function access the fileids of the corpus
        """
        if not any(key.startswith('fileids') for key in kwargs.keys()):
            kwargs['fileids'] = None

        if not any(key.startswith('categories') for key in kwargs.keys()):
            kwargs['categories'] = None

        fileids = self.corpus.resolve(**kwargs)
        if fileids:
            return fileids
        return self.corpus.fileids()

    @staticmethod
    def tokenize(text) -> list:
        """
        Segments, tokenizes, and tags document text in the corpus. Returns
        text, which is a list of sentences, in turn is a lists of part of
        speech tagged words (token, tag).

        Parameters
        ----------
        text : str
            single string containing the text to be tokenized

        Returns
        -------
            a list of formatted text

        """
        try:
            tokenized_text = [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(text)
            ]
        except KeyError:
            tokenized_text = [[('', '')]]

        try:
            if type(tokenized_text[0]) is tuple:
                return [tokenized_text]
            else:
                return tokenized_text
        except IndexError:
            return [[('', '')]]

    def process(self, fileid, fields=None):
        """
        For a single file does the following preprocessing work:
            1. Get the location of the document
            2. Generate a structured text list,
            3. Append the new structured text to the existing document
            4. Append filename to document
            5. Writes the document as a pickle to the target location.
            6. Clean up the document
        This method is called multiple times from the transform runner.
        Parameters
        ----------
        fields : list
            list of field identifiers
        fileid : str
            the file id to be processed

        Returns
        -------
            None

        """
        if fields is None:
            fields = ['dc:title', 'dc:description']

        # 1. Get the location to store file
        file_path = os.path.join(self.target, fileid)

        # iterate over fields
        document = self.corpus.read_single(fileid)
        # 2 Generate structured form of text
        for field in fields:
            try:
                text = document[field]
            except KeyError:
                text = ''

            tokenized_text = self.tokenize(text)

            # 3 Append structured form of text to document
            new_field = "processed:{}".format(field)
            document[new_field] = tokenized_text

        # 4. Append filename to document
        document['file_name'] = fileid

        # 5. Writes the document as a pickle to the target location.
        with open(file_path, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

        # 6. Clean up the document
        del document

    def transform(self, **kwargs):
        """
        Take an existing corpus and transform it such that it contains more
        suitably formatted data.

        process each file in the corpus
        Parameters
        ----------
        Returns
        -------
            None

        """
        start = time.time()
        # serial file processing

        # create the target folders

        for cat in self.corpus.categories():
            Utils.make_folder('{}/{}'.format(self.target, cat))

        for filename in tqdm(self.fileids(**kwargs),
                         desc="Processing corpus"):
            self.process(filename, fields=None)

        end = time.time()
        print("Time to process {}seconds".format(end - start))


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader
    from pprint import PrettyPrinter

    root = "Corpus/Split_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusCorpusReader(root=root)
    target = "Corpus/Processed_corpus/"

    # gen = corpus.docs()
    # aa = next(gen)
    # pp = PrettyPrinter(indent=4)
    # pp.pprint(aa['dc:title'])
    #
    # text = aa['dc:title']
    # bb = [pos_tag(wordpunct_tokenize(sent)) for sent in sent_tokenize(text)]

    processor = ScopusCorpusProcessor(corpus=corpus, target=target)
    processor.transform()

    corpus2 = Elsevier_Corpus_Reader.ScopusCorpusReader(root=target)
    gen2 = corpus2.docs()
    aa2 = next(gen2)
    pp = PrettyPrinter(indent=4)
    pp.pprint(aa2['processed:dc:title'])
    pp.pprint(aa2['processed:dc:description'])
    pp.pprint(aa2['file_name'])
