
from Ingestor import Elsevier_Ingestor
from PreProcessor import Elsivier_Pre_Processor
from CorpusReader import Elsevier_Corpus_Reader


def download_corpus():
    """
    download a all papers for a given search term for a given year.
    Returns
    -------

    """
    search_terms = ['soft robot']
    dates = (1986, 1987)

    builder = Elsevier_Ingestor.ElsevierIngestionEngine(
        search_terms=search_terms,
        file_path="Corpus/Raw_corpus/",
        dates=dates,
        home=False,
        # database='scopus',
        database='science_direct',
        batch_size=25)

    builder.build_corpus()


def preprocess_corpus():
    root = "Corpus/Raw_corpus/"
    target = "Corpus/Processed_corpus/"

    corpus = Elsivier_Pre_Processor.PickledCorpusPreProcesor(root=root,
                                                           target=target)
    corpus.refactor_corpus()


def read_corpus():
    """
    Read the contains the corpus reader.
    Returns
    -------

    """
    corp = Elsevier_Corpus_Reader.PickledCorpusReader("Corpus/")
    corp.fileids()
    corp.categories()
    for _ in corp.titles():
        print(_)

if __name__ == '__main__':
    # download_corpus()
    # preprocess_corpus()
    corp = Elsevier_Corpus_Reader.PickledCorpusReader(
        "Corpus/Processed_corpus/")