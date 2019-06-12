# /home/martin/Documents/RESEARCH/fictional-barnacle/CorpusProcessingTools/
"""
Plotting_Tools.py

@author: martinventer
@date: 2019-06-13

Reads the pre-processed Corpus data and generates bibliometric plots
"""

from CorpusReader import Elsevier_Corpus_Reader
from collections import Counter
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt


def document_feature_counter(path,
                             feature='pub_date',
                             sort=False,
                             how='count',
                             **kwargs) -> dict:
    """
    utility for counting the number of instances observed for a given feature
    in the document meta data
    Parameters
    ----------
    path: str
        path to the corpus
    feature: str
        feature to be counted
            'pub_date' - date of publication
            'pub_type' - type of document, eg. Article, Review ...
            'publication' - journal in which the document is published
    sort: bool
        should the output dictionary be sorted or not
    how: str
        if the output should be sorted, how should it be sorted
        'class' - sorted by the class, requires a sortable class, eg. dates
        'count' - sorted by the number of counts of a class
    kwargs:
        optional arguments that can be piped through to an underlying corpus
        reader method.

    Returns
    -------
        dict like object, either a Counter object or an OrderedDict
    """
    corp = Elsevier_Corpus_Reader.PickledCorpusReader(path)
    feature_map = {'pub_date': corp.pub_date,
                   'pub_type': corp.pub_type,
                   'publication': corp.publication,
                   'author_count': corp.author_count}
    sort_how_map = {'class': 0,
                    'count': 1}
    if kwargs:
        data = Counter(feature_map[feature](**kwargs))
    else:
        data = Counter(feature_map[feature]())
    if not sort:
        return data
    else:
        sorted_data = sorted(data.items(), key=lambda kv: kv[sort_how_map[
            how]])
        return OrderedDict(sorted_data)


def bar_plot(data_dict):
    """
    General bar plotting tool for the dict like output of
    document_feature_counter()
    Parameters
    ----------
    data_dict: dict like
        output from     document_feature_counter()

    Returns
    -------
        None
    """
    sns.barplot(list(data_dict.keys()), list(data_dict.values()))
    plt.show()


def plot_documents_per_pub_date(corpus_path,
                                how='count',
                                form='year'):
    """
    wrapper for document_feature_counter() and bar_plot() applied to the
    documents per publication date
    Parameters
    ----------
        how : str
            default sorting by the value count
        form :
            default is to look only at the year of publication
        corpus_path : object
            path to the preprocessed corpus

    Returns
    -------
        None
    """
    data = document_feature_counter(path=corpus_path,
                                    feature='pub_date',
                                    sort=True,
                                    how=how,
                                    form=form)
    bar_plot(data)


def plot_documents_per_pub_type(corpus_path,):
    """
    wrapper for document_feature_counter() and bar_plot() applied to the
    documents per publication type
    Parameters
    ----------
        corpus_path : object
            path to the preprocessed corpus

    Returns
    -------
        None
    """
    data = document_feature_counter(path=corpus_path,
                                    feature='pub_type',
                                    sort=True,
                                    how='count')
    bar_plot(data)


def plot_distribution_of_docs_in_publications(corpus_path):
    """
    wrapper for document_feature_counter() and bar_plot() applied to the
    distribution of papers by journal
    Parameters
    ----------
        corpus_path : object
            path to the preprocessed corpus

    Returns
    -------
        None
    """
    data = document_feature_counter(path=corpus_path,
                                    feature='publication',
                                    sort=True,
                                    how='count')
    bar_plot(Counter(data.values()))


def plot_distribution_authors_per_document(corpus_path):
    """
    wrapper for document_feature_counter() and bar_plot() applied to the
    distribution of authors per document
    Parameters
    ----------
        corpus_path : object
            path to the preprocessed corpus

    Returns
    -------
        None
    """
    data = document_feature_counter(path=corpus_path,
                                    feature='author_count',
                                    sort=True,
                                    how='count')
    # return data
    bar_plot(data)


if __name__ == "__main__":
    # plot_documents_per_pub_date("Corpus/Processed_corpus/")
    # plot_documents_per_pub_type("Corpus/Processed_corpus/")
    # plot_distribution_of_docs_in_publications("Corpus/Processed_corpus/")
    test = plot_distribution_authors_per_document("Corpus/Processed_corpus/")