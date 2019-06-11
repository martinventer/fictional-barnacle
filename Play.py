
from Ingestor import Elsevier_Ingestor
from CorpusReader import Elsevier_Corpus_Reader
from PreProcessor import Elsivier_Pre_Processor
import pickle
from tqdm import tqdm


import hashlib
# print(hashlib.md5("whatever your string is".encode('utf-8')).hexdigest())

def download_corpus():

    search_terms = ['soft robot']
    dates = (1986, 1987)


    builder = Elsevier_Ingestor.ElsevierIngestionEngine(search_terms=search_terms,
                                                    file_path="Raw_corpus/",
                                                    dates=dates,
                                                    home=False,
                                                    # database='scopus',
                                                    database='science_direct',
                                                    batch_size=25)

    builder.build_corpus()


def compile_corpus():
    root = "Raw_corpus/"
    target = "Corpus/"

    corp = Elsivier_Pre_Processor.PickledCorpusPreProcesor(root=root,
                                                           target=target)
    corp.refactor_corpus()

def read_corpus():
    corp = Elsevier_Corpus_Reader.PickledCorpusReader("Raw_corpus/")
    corp.fileids()
    corp.categories()

    # for _ in corp.titles():
    #     print(_)

if __name__ == '__main__':
    # download_corpus()
    compile_corpus()
    corp = Elsevier_Corpus_Reader.PickledCorpusReader("Corpus/")
    corp.fileids()
    # corp.categories()
    # doc = next(corp.docs('soft robot/1986.pickle'))
