#! envs/fictional-barnacle/bin/python3.6
"""
Text_Visualization.py

@author: martinventer
@date: 2019-07-08

Tools for Visualizing Text related data
"""

from CorpusProcessingTools import Corpus_Vectorizer, Context_Extraction

import matplotlib.pyplot as plt
from yellowbrick.text.freqdist import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter

import itertools
import networkx as nx

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

from networkx.drawing.nx_agraph import graphviz_layout


def plot_term_frequency(corpus, n_terms=50) -> None:
    """
    plot the term frequencies of the n_terms most common terms in a corpus. a raw
    corpus on [[(token, tag)]], is cleaned and plotted
    Parameters
    ----------
    corpus
        a coupus object or generator.
    n_terms : int
        the number of terms you are interested in

    Returns
    -------

    """
    corpus_cleaner = Corpus_Vectorizer.TextNormalizer()
    clean_corpus = corpus_cleaner.fit_transform(corpus)

    vectorizer = Corpus_Vectorizer.Text2FrequencyVector()
    docs = vectorizer.fit_transform(clean_corpus)
    features = vectorizer.get_feature_names()
    print(docs.shape)
    print(len(features))
    print(features[:4])

    visualizer = FreqDistVisualizer(features=features, n=n_terms)
    visualizer.fit(docs)
    visualizer.poof()


def plot_keyphrase_frequency(corpus, n_terms=50) -> None:
    """
    plot the keyphrase frequencies of the n_terms most common terms in a corpus. a raw
    corpus on [[(token, tag)]], is cleaned and plotted
    Parameters
    ----------
    corpus
        a coupus object or generator.
    n_terms : int
        the number of terms you are interested in

    Returns
    -------

    """
    entity_extractor = Context_Extraction.KeyphraseExtractorS()
    entities = list(entity_extractor.fit_transform(corpus))

    vectorizer = Corpus_Vectorizer.Text2wordCountVector()
    docs = vectorizer.fit_transform(entities)
    features = vectorizer.get_feature_names()

    visualizer = FreqDistVisualizer(features=features, n=n_terms)
    visualizer.fit(docs)
    visualizer.poof()


def iter_flatten(iterable):
    """
    A recursive iterator to flatten nested lists
    Parameters
    ----------
    iterable

    Returns
    -------
        yields next item
            to get a flat the nested list 'a'
                a = [i for i in iter_flatten(a)]
    """
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in iter_flatten(e):
                yield f
        else:
            yield e


def most_common_terms(observations, n_terms=50):
    """
    returns the most common terms in a set documents from a corpus.
    Parameters
    ----------
    observations : list
        nested list of obsercations
    n_terms : int
        the number of terms that should be included
    Returns
    -------
        a tuple pair list
    """
    flat_observations = [i for i in iter_flatten(observations)]
    word_count = Counter(flat_observations)

    return zip(*word_count.most_common(n_terms))


def cooccurrence(observations, terms,
                 occurance_depth=2,
                 minimum_occurance=1) -> dict:
    """
    assembles a dictionary containing instances of co-occurance of terms in a
    list of listed observations
    Parameters
    ----------
    minimum_occurance : int
        minimum number of occurance to register a connection.
    occurance_depth : int
        depth of the co-occurance evaluation. default is 2 so the occurance
        of a pair of terms will be tracked. If the value is increased a
        higher order occuance will be required.
    observations : corpus view
        document text data in the form [paragraph[sentance(token, tag)]]
    terms : list
        terms to consider in the co-occurance matrix
    Returns
    -------

    """
    # create a list of all possible paris of terms
    possible_pairs = list(itertools.combinations(terms, occurance_depth))

    # initialise a dictionary containing an entry for each pair of words
    default_count = 0
    cooccurring = dict.fromkeys(possible_pairs, default_count)

    # incriment possible pairs for each occurance
    for observation in observations:
        for pair in possible_pairs:
            if pair[0] in observation and pair[1] in observation:
                cooccurring[pair] += 1

    # remove cases where the co-occurance is lest than a set mimimum

    return cooccurring


def plot_term_coocurrance(docs, n_terms=30) -> None:
    """
    plot a co-occurrence network given view of a corpus.
    Parameters
    ----------
    docs : generator
        document text data in the form [paragraph[sentance(token, tag)]]
    n_terms : int
        the number of terms that should be included
    Returns
    -------

    """
    normalizer = Corpus_Vectorizer.TextNormalizer()
    normed = normalizer.transform(docs)

    fig, ax = plt.subplots(1, figsize=(15, 12))
    g = nx.Graph()

    # create nodes for each term
    frequent_terms, term_count = most_common_terms(normed, n_terms=n_terms)
    g.add_nodes_from(list(set(frequent_terms)))

    # create edges for each pair of connected terms
    pairs = cooccurrence(observations=normed, terms=frequent_terms)
    for pair, wgt in pairs.items():
        if wgt > 0:
            g.add_edge(pair[0], pair[1], weight=wgt)

    node_options = {'alpha': 0.5}
    # assign node scale
    if True:
        node_size = []
        node_scale = 100
        for node in g:
            node_size.append(node_scale ** 1.1)
        node_options["node_size"] = node_size

    # adjust the node layout style
    pos = graphviz_layout(g, prog="twopi")
    nx.draw_networkx_nodes(g, pos=pos, **node_options)

    nx.draw_networkx_nodes(g, pos=pos)
    if True:
        node_names = {}
        for node in g:
            node_names[node] = node  # .split()[-1]
        nx.draw_networkx_labels(g, pos,
                                labels=node_names,
                                font_size=8)

    # draw in the edges
    nx.draw_networkx_edges(g, pos=pos,
                           alpha=0.2)

    # nx.draw(g, pos=pos, ax=ax)
    # nx.draw(g, pos, node_color="gold", node_size=50, edgelist=edges,
    #         width=.5, edge_color="orange", with_labels=True, font_size=12)

    # plot the figure
    ax.axis('off')
    fig.tight_layout()
    plt.show()


def matrix(corpus, terms, fileids=None):
    # # get the most common words in the corpus
    # frequent_terms, _ = most_common_terms(corpus, n_terms=n_terms, fileids=fileids)
    # for term in frequent_terms: print(term)

    # run through each document title and invriment coccurance of terms
    docs = corpus.title_tagged(fileids=fileids)
    normalizer = Corpus_Vectorizer.TextNormalizer()
    normed = normalizer.transform(docs)

    mtx = []
    for first in terms:
        row = []
        for second in terms:
            count = 0
            for doc in normed:
                if first in doc and second in doc:
                    count += 1
            row.append(count)
        mtx.append(row)
    return mtx



def plot_term_coocurrance_matrix(corpus, n=30, fileids=None):
    ##############################################
    # Plot a Co-Occurrence Matrix
    ##############################################
    # First make the matrices
    frequent_terms, _ = most_common_terms(corpus, n_terms=n, fileids=fileids)
    # By frequency
    mtx = matrix(corpus=corpus, terms=frequent_terms, fileids=fileids)

    # Now create the plots
    fig, ax = plt.subplots(figsize=(9, 6))

    x_tick_marks = np.arange(n)
    y_tick_marks = np.arange(n)

    ax.set_xticks(x_tick_marks)
    ax.set_yticks(y_tick_marks)
    ax.set_xticklabels(frequent_terms, fontsize=8, rotation=90)
    ax.set_yticklabels(frequent_terms, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_xlabel("By Frequency")
    plt.imshow(mtx, norm=LogNorm(), interpolation='nearest', cmap='YlOrBr')

    plt.show()


def plot_term_occurance_over_time(corpus, n=30, fileids=None):
    # #############################################
    # Plot mentions of characters through over time
    # #############################################
    frequent_terms, _ = most_common_terms(corpus, n_terms=n, fileids=fileids)
    docs = corpus.title_tagged(fileids=subset)

    normalizer = Corpus_Vectorizer.TextNormalizer()
    normed = normalizer.transform(docs)

    dates = corpus.pub_date(form='year', fileids=subset)

    x, y = [], []
    for doc, date in zip(normed, dates):
        for i, term in enumerate(frequent_terms):
            if term in doc:
                x.append(date)
                y.append(i)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(x, y, "*")

    plt.yticks(list(range(len(frequent_terms))), frequent_terms, size=8)
    plt.ylim(-1, len(frequent_terms))
    plt.title("Character Mentions in the Wizard of Oz")
    plt.show()


def plot_tsne_clusters(corpus, fileids=None, labels=None):
    from yellowbrick.text import TSNEVisualizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    words = corpus.title_tagged(fileids=fileids)
    normalizer = Corpus_Vectorizer.TextNormalizer()
    normed = (sent for title in normalizer.transform(words) for sent in title)
    # normed = (dd for dd in normalizer.transform(observations))
    tfidf = TfidfVectorizer()
    procd = tfidf.fit_transform(normed)

    tsne = TSNEVisualizer()
    if labels is None:
        tsne.fit(procd)
    else:
        tsne.fit(procd, ["c{}".format(c) for c in labels])
    tsne.poof()


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader
    # from CorpusProcessingTools import Corpus_Vectorizer, Corpus_Cluster
    from sklearn.pipeline import Pipeline

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)
    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 10, shuffle=False)
    subset_fileids = next(loader.fileids(test=True))

    # --------------------------------------------------------------------------
    # plot_term_frequency
    # --------------------------------------------------------------------------
    # plot_term_frequency(corpus.title_tagged(fileids=subset_fileids))
    # plot_term_frequency(corpus.description_tagged(fileids=subset_fileids))

    # --------------------------------------------------------------------------
    # plot_term_frequency
    # --------------------------------------------------------------------------
    # plot_keyphrase_frequency(corpus.title_tagged(fileids=subset_fileids))
    # plot_keyphrase_frequency(corpus.description_tagged(fileids=subset_fileids))

    # --------------------------------------------------------------------------
    # plot_term_coocurrance
    # --------------------------------------------------------------------------
    plot_term_coocurrance(corpus.title_tagged(fileids=subset_fileids),
                          n_terms=50)
    # plot_term_coocurrance_matrix(corpus, n_terms=30, fileids=subset_fileids)
    # plot_term_occurance_over_time(corpus, n_terms=30, fileids=subset_fileids)
    # plot_tsne_clusters(corpus, fileids=subset)

    # # with clusters
    # model = Pipeline([A
    #     ("norm", Corpus_Vectorizer.TitleNormalizer()),
    #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
    #     ('clusters', Corpus_Cluster.MiniBatchKMeansClusters(k=3))
    # ])
    #
    # observations = corpus.title_tagged(fileids=subset)
    # clusters = model.fit_transform(observations)
    #
    # plot_tsne_clusters(corpus, fileids=subset, labels=clusters)

