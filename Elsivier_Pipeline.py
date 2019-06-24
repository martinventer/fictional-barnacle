
from Ingestor import Elsivier_Ingestor
from CorpusProcessingTools import Elsivier_Corpus_Pre_Processor
from CorpusReader import Elsevier_Corpus_Reader
from CorpusProcessingTools import Plotting_Tools, Author_Networks


def download_corpus():
    """
    download a all papers for a given search term for a given year.
    Returns
    -------

    """
    search_terms = ['soft robot']
    dates = (1950, 2021)

    builder = Elsivier_Ingestor.ScopusIngestionEngine(
        search_terms=search_terms,
        file_path="Corpus/Raw_corpus/",
        dates=dates,
        home=False,
        batch_size=25)

    builder.build_corpus()


def reformat_corpus():
    root = "Corpus/Raw_corpus/"
    target = "Corpus/Processed_corpus/"

    corpus = Elsivier_Corpus_Pre_Processor.PickledCorpusRefactor(root=root,
                                                                 target=target)
    corpus.refactor_corpus()

def process_corpus():
    corp = Elsevier_Corpus_Reader.ScopusRawCorpusReader(
            "Corpus/Processed_corpus/")

    formatter = Elsivier_Corpus_Pre_Processor.PickledCorpusPreProcessor(corp)

    formatter.transform()


def plot_features():
    AN = Author_Networks.AuthorNetworks("Corpus/Processed_corpus/")
    # AN.plot_co_author_network(categories='soft robot/2000')
    AN.co_author_network_bokeh_better(categories=['soft robot/2000',
                                                  'soft robot/2001',
                                                  'soft robot/2002'])


if __name__ == '__main__':
    # step 1: download the raw corpus from elsivier
    # download_corpus()

    # step 2: reformat the corpus for faster manipulation
    # reformat_corpus()

    # step 3: reformat the corpus for faster manipulation
    # process_corpus()

    # step 4: load the corpus reader
    corp = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    # step 5: plot author connectivity
    # plot_features()

