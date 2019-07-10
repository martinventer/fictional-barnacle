
from Ingestor import Elsivier_Ingestor, Corpus_Pre_Processor
from CorpusReaders import Elsevier_Corpus_Reader
from PlottingTools import Author_Networks


def download_corpus():
    """
    download a all papers for a given search term for a given year.
    Returns
    -------

    """
    builder = Elsivier_Ingestor.ScopusIngestionEngine(
        file_path="Corpus/Raw_corpus/",
        home=False,
        batch_size=25)

    builder.build_corpus(search_terms=['soft robot'],
                         dates=(1998, 1999))


def reformat_corpus():
    root = "Corpus/Raw_corpus/"
    target = "Corpus/Split_corpus/"

    corpus = Elsevier_Corpus_Reader.RawCorpusReader(root=root)

    Corpus_Pre_Processor.split_corpus(corpus=corpus, target=target)


def process_corpus():
    corp = Elsevier_Corpus_Reader.ScopusCorpusReader(
            "Corpus/Processed_corpus/")

    formatter = Corpus_Pre_Processor.ScopusCorpusProcessor(corp)

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
    # corp = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
    #     "Corpus/Processed_corpus/")

    # step 5: plot author connectivity
    # plot_features()
