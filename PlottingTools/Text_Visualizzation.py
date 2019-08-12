#! envs/fictional-barnacle/bin/python3.6
"""
Text_Visualization.py

@author: martinventer
@date: 2019-07-08

Tools for Visualizing Text related data
"""

from CorpusProcessingTools import Corpus_Vectorizer, Context_Extraction

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from yellowbrick.text.freqdist import FreqDistVisualizer
from networkx.drawing.nx_agraph import graphviz_layout

from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter
import itertools
import networkx as nx

import numpy as np
from scipy.sparse import csr_matrix


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


def cooccurrence_dict(observations, terms,
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
    cooccurring = {k: v for k, v in cooccurring.items() if v > minimum_occurance}

    return cooccurring


def plot_term_coocurrance_network(docs,
                                  n_terms=30,
                                  **kwargs) -> None:
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
    term_counts = dict(zip(frequent_terms, term_count))
    g.add_nodes_from(list(set(frequent_terms)))

    # create edges for each pair of connected terms
    pairs = cooccurrence_dict(observations=normed,
                              terms=frequent_terms,
                              **kwargs)
    edge_width = []
    edge_scale = 0.1
    for pair, wgt in pairs.items():
        g.add_edge(pair[0], pair[1])
        edge_width.append(edge_scale * wgt)

    # set layout style
    # pos = graphviz_layout(g, prog="dot") # Slow
    # pos = graphviz_layout(g, prog="neato")
    # pos = graphviz_layout(g, prog="fdp")
    # pos = graphviz_layout(g, prog="sfdp")
    # pos = graphviz_layout(g, prog="twopi")
    # pos = graphviz_layout(g, prog="circo") # Slow
    # pos = nx.spring_layout(g)
    # pos = nx.spectral_layout(g)
    pos = nx.kamada_kawai_layout(g)

    node_options = {'alpha': 0.5}
    # assign node scale to a multiple of its term count.
    if True:
        node_size = []
        node_scale = 10.0
        for node in g:
            node_size.append(node_scale * term_counts[node])
        node_options["node_size"] = node_size

    # draw the nodes
    nx.draw_networkx_nodes(g, pos=pos, **node_options)

    if True:
        node_names = {}
        for node in g:
            node_names[node] = node  # .split()[-1]
        nx.draw_networkx_labels(g, pos,
                                labels=node_names,
                                font_size=8)

    # draw in the edges
    nx.draw_networkx_edges(g, pos=pos,
                           alpha=0.2,
                           width=edge_width)

    # plot the figure
    ax.axis('off')
    fig.tight_layout()
    plt.show()


def create_co_occurences_matrix(allowed_words, documents):
    """
    creates a co-occurance matrix given a list off search terms and documents
    Parameters
    ----------
    allowed_words : List
        list of terms to find
    documents : List of lists
        text from ducument corpus

    Returns
    -------
        A fully dense numpy array for the co-occurance matrix
        a list of word ids

    """
    # make sure that the input text is in a flat list.
    allowed_words = [i for i in iter_flatten(allowed_words)]

    # convert the word list to a distionary of terms with unique intiger ids
    word_to_id = dict(zip(allowed_words, list(range(len(allowed_words)))))

    # create a sorted document  lookup
    documents_as_ids = [
        np.sort([word_to_id[w]
                 for w in doc if w in word_to_id]).astype('uint32')
        for doc in documents]

    # Create a 2D array of terms vs terms
    row_ind, col_ind = zip(
        *itertools.chain(*[[(i, w) for w in doc]
                           for i, doc in enumerate(documents_as_ids)]))

    # Generate an array of term counts per document
    # use unsigned int for better memory utilization
    data = np.ones(len(row_ind), dtype='uint32')
    max_word_id = max(itertools.chain(*documents_as_ids)) + 1
    docs_words_matrix = csr_matrix((data,
                                    (row_ind, col_ind)),
                                   shape=(len(documents_as_ids),
                                          max_word_id))

    # multiplying docs_words_matrix with its transpose matrix to generate the
    # co-occurences matrix
    words_cooc_matrix = docs_words_matrix.T * docs_words_matrix
    words_cooc_matrix.setdiag(0)
    return words_cooc_matrix.toarray(), word_to_id


def plot_term_coocurrance_matrix(docs, n_terms=30, **kwargs) -> None:
    """
    plot a co-occurrence matrix given view of a corpus.
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

    # create nodes for each term
    frequent_terms, term_count = most_common_terms(normed, n_terms=n_terms)

    # By frequency
    mtx, ids = create_co_occurences_matrix(frequent_terms, normed)
    print(mtx, ids)

    # Now create the plots
    fig, ax = plt.subplots(figsize=(9, 6))

    x_tick_marks = np.arange(n_terms)
    y_tick_marks = np.arange(n_terms)

    ax.set_xticks(x_tick_marks)
    ax.set_yticks(y_tick_marks)
    ax.set_xticklabels(frequent_terms, fontsize=8, rotation=90)
    ax.set_yticklabels(frequent_terms, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_xlabel("By Frequency")
    plt.imshow(mtx.tolist(), norm=LogNorm(), interpolation='nearest',
               cmap='YlOrBr')
    plt.show()


def plot_term_occurance_over_time(docs, dates, n_terms=30) -> None:
    """
    Plots a tick for each occurance of a term in a data set plotted over time
    Parameters
    ----------
    docs : Corpus view
        a corpus for a text field
    dates : Corpus Vies
        a view of the date of publication for each document
    n_terms : int
        The number of desired common terms.

    Returns
    -------

    """
    # preprocess text data
    normalizer = Corpus_Vectorizer.TextNormalizer()
    normed = normalizer.transform(docs)

    # extract the most common terms in the documnet list
    frequent_terms, term_count = most_common_terms(normed, n_terms=n_terms)

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
    plt.title("Occurance of the {0} most frequent terms".format(n_terms))
    plt.show()


def plot_term_tsne_clusters(docs, labels=None):
    from yellowbrick.text import TSNEVisualizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    # words = corpus.title_tagged(fileids=fileids)
    # normalizer = Corpus_Vectorizer.TextNormalizer()
    # normed = (sent for title in normalizer.transform(words) for sent in title)

    # preprocess text data
    normalizer = Corpus_Vectorizer.TextNormalizer()
    normed = normalizer.transform(docs)

    # unbundle sentences
    normed = (sent for doc in normed for sent in doc)

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
    # plot_term_frequency(
    #     corpus.title_tagged(fileids=subset_fileids))
    # plot_term_frequency(
    #     corpus.description_tagged(fileids=subset_fileids))

    # --------------------------------------------------------------------------
    # plot_keyphrase_frequency
    # --------------------------------------------------------------------------
    # plot_keyphrase_frequency(
    #     corpus.title_tagged(fileids=subset_fileids))
    # plot_keyphrase_frequency(
    #     corpus.description_tagged(fileids=subset_fileids))

    # --------------------------------------------------------------------------
    # plot_term_coocurrance_network
    # --------------------------------------------------------------------------
    # plot_term_coocurrance_network(
    #     corpus.title_tagged(fileids=subset_fileids),
    #     n_terms=50,
    #     minimum_occurance=10)

    # plot_term_coocurrance_network(
    #     corpus.description_tagged(fileids=subset_fileids),
    #     n_terms=50,
    #     minimum_occurance=10)

    # --------------------------------------------------------------------------
    # plot_term_coocurrance_matrix
    # --------------------------------------------------------------------------
    # plot_term_coocurrance_matrix(
    #     corpus.title_tagged(fileids=subset_fileids),
    #     n_terms=30)

    # plot_term_coocurrance_matrix(
    #     corpus.description_tagged(fileids=subset_fileids),
    #     n_terms=30)

    # --------------------------------------------------------------------------
    # plot_term_occurance_over_time
    # --------------------------------------------------------------------------
    # plot_term_occurance_over_time(
    #     corpus.title_tagged(fileids=subset_fileids),
    #     corpus.publication_date(fileids=subset_fileids),
    #     n_terms=30)

    # plot_term_occurance_over_time(
    #     corpus.description_tagged(fileids=subset_fileids),
    #     corpus.publication_date(fileids=subset_fileids),
    #     n_terms=30)

    # --------------------------------------------------------------------------
    # plot_term_tsne_clusters
    # --------------------------------------------------------------------------
    plot_term_tsne_clusters(
        corpus.title_tagged(fileids=subset_fileids))

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

