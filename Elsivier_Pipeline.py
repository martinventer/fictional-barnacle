from CorpusReaders import Elsevier_Corpus_Reader, Corpus_Pre_Processor, \
    Elsivier_Ingestor
from PlottingTools import Author_Networks


def download_corpus() -> None:
    """
    download a all papers for a given search term for a given year.
    Returns
    -------
        None
    """
    builder = Elsivier_Ingestor.ScopusIngestionEngine(
        file_path="Corpus/Raw_corpus/",
        home=False,
        batch_size=25)

    builder.build_corpus(search_terms=['soft robot'],
                         dates=(1950, 2021))


def refactor_corpus() -> None:
    """
    Read the raw corpus and refactor the collections from a single file per
    year to a single file per document.
    Returns
    -------
        None
    """
    root = "Corpus/Raw_corpus/"
    target = "Corpus/Split_corpus/"

    corpus = Elsevier_Corpus_Reader.RawCorpusReader(root=root)

    Corpus_Pre_Processor.split_corpus(corpus=corpus, target=target)


def preprocess_corpus() -> None:
    """
    processes and refomats both text and meta data
    Returns
    -------
        None
    """
    corp = Elsevier_Corpus_Reader.ScopusCorpusReader(
            "Corpus/Split_corpus/")

    formatter = Corpus_Pre_Processor.ScopusCorpusProcessor(corp)

    formatter.transform()


if __name__ == '__main__':
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    # ==========================================================================
    # step 1: download the raw corpus from elsivier
    #   search term 'Soft robot'
    #   Date Range between 1950 and 2021
    #   Downloaded on 26 August
    # ==========================================================================
    if False:
        download_corpus()

    # ==========================================================================
    # step 2: refactor the corpus
    #   The corpus is downloaded as collections containing all publications
    #   within a given year. This step splits these collections into
    #   individual documents that can be accessed independently.
    # ==========================================================================
    if False:
        refactor_corpus()

    # ==========================================================================
    # step 3: Pre process the corpus to clean and format the data.
    #   tokenize text fields
    #   tokenize metadata fields
    #   add file name to metadata
    # ==========================================================================
    if False:
        preprocess_corpus()

    # ==========================================================================
    # step 4: Preliminary exploration
    #   How many documents does the corpus contain
    # ==========================================================================
    if True:
        root = "Corpus/Split_corpus/"
        corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(root)
        pp.pprint(corpus.describe())


