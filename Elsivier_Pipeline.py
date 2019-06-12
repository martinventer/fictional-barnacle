
from Ingestor import Elsevier_Ingestor
from CorpusProcessingTools import Elsivier_Corpus_Pre_Processor
from CorpusReader import Elsevier_Corpus_Reader
from CorpusProcessingTools import Plotting_Tools


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

    corpus = Elsivier_Corpus_Pre_Processor.PickledCorpusPreProcesor(root=root,
                                                                    target=target)
    corpus.refactor_corpus()


def plot_features():
    Plotting_Tools.plot_documents_per_pub_date("Corpus/Processed_corpus/")
    # Plotting_Tools.plot_documents_per_pub_type("Corpus/Processed_corpus/")
    # Plotting_Tools.plot_distribution_of_docs_in_publications(
    #     "Corpus/Processed_corpus/")


if __name__ == '__main__':
    # download_corpus()
    # preprocess_corpus()
    # plot_features()

    # create a corpus reader object
    corp = Elsevier_Corpus_Reader.PickledCorpusReader(
        "Corpus/Processed_corpus/")




