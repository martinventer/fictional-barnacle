#! envs/fictional-barnacle/bin/python3.6
"""
Text_Visualization.py

@author: martinventer
@date: 2019-07-08

Tools for Visualizing Text related data
"""

from CorpusProcessingTools import Corpus_Vectorizer

import matplotlib.pyplot as plt
from yellowbrick.text.freqdist import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter

import json
import codecs
import itertools
import networkx as nx

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from nltk import sent_tokenize, word_tokenize


def plot_term_frequency(subset):
    ##############################################
    # Visualize frequency distribution of top 50 tokens
    ##############################################
    vectorizer = CountVectorizer()
    docs = vectorizer.fit_transform(corpus.title_words(fileids=subset))
    features = vectorizer.get_feature_names()


    visualizer = FreqDistVisualizer(features)
    visualizer.fit(docs)
    visualizer.poof()


def plot_term_frequency_no_stop(subset):
    ##############################################
    # Visualize stopwords removal
    ##############################################
    vectorizer = CountVectorizer(stop_words='english')
    docs = vectorizer.fit_transform(corpus.title_words(fileids=subset))
    features = vectorizer.get_feature_names()

    visualizer = FreqDistVisualizer(features)
    visualizer.fit(docs)
    visualizer.poof()


# def cooccurrence(text, cast):
#     possible_pairs = list(itertools.combinations(cast, 2))
#     cooccurring = dict.fromkeys(possible_pairs, 0)
#     for title, chapter in text['chapters'].items():
#         for sent in sent_tokenize(chapter):
#             for pair in possible_pairs:
#                 if pair[0] in sent and pair[1] in sent:
#                     cooccurring[pair] += 1
#     return cooccurring


def most_common_terms(corpus, n=50, fileids=None):
    # get the most common words in the corpus
    words = corpus.title_tagged(fileids=fileids)
    normalizer = Corpus_Vectorizer.TextNormalizer()
    normed = (sent for title in normalizer.transform(words) for sent in title)
    word_count = Counter(normed)

    return zip(*word_count.most_common(n))



def cooccurrence(corpus, terms, fileids=None):
     # get the possible paris from the most common words
    possible_pairs = list(itertools.combinations(terms, 2))

    # create an empty dictionary containing an entry for each pair of words
    cooccurring = dict.fromkeys(possible_pairs, 0)

    # run through each document title and invriment coccurance of terms
    docs = corpus.title_tagged(fileids=fileids)
    normalizer = Corpus_Vectorizer.TextNormalizer()
    normed = (dd for dd in normalizer.transform(docs))
    for doc in normed:
        for pair in possible_pairs:
            if pair[0] in doc and pair[1] in doc:
                cooccurring[pair] += 1
    return cooccurring


def plot_term_coocurrance(corpus, n=30, fileids=None):
    ##############################################
    # Build a NetworkX Graph
    ##############################################

    frequent_terms, _ = most_common_terms(corpus, n=n, fileids=fileids)

    G = nx.Graph()
    G.name = "The Social Network of Oz"
    pairs = cooccurrence(corpus=corpus, terms=frequent_terms, fileids=fileids)
    for pair, wgt in pairs.items():
        if wgt > 0:
            G.add_edge(pair[0], pair[1], weight=wgt)
    # Make Dorothy the center
    D = nx.ego_graph(G, "robot")
    edges, weights = zip(*nx.get_edge_attributes(D, "weight").items())
    # Push nodes away that are less related to Dorothy
    pos = nx.spring_layout(D, k=.5, iterations=40)
    nx.draw(D, pos, node_color="gold", node_size=50, edgelist=edges,
            width=.5, edge_color="orange", with_labels=True, font_size=12)
    plt.show()



def matrix(corpus, terms, fileids=None):
    # # get the most common words in the corpus
    # frequent_terms, _ = most_common_terms(corpus, n=n, fileids=fileids)
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
    frequent_terms, _ = most_common_terms(corpus, n=n, fileids=fileids)
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
    frequent_terms, _ = most_common_terms(corpus, n=n, fileids=fileids)
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
    # normed = (dd for dd in normalizer.transform(docs))
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
    from CorpusProcessingTools import Corpus_Vectorizer, Corpus_Cluster
    from sklearn.pipeline import Pipeline


    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 12, shuffle=False)
    subset = next(loader.fileids(test=True))

    # plot_term_frequency(subset)
    # plot_term_frequency_no_stop(subset)
    # plot_term_coocurrance(corpus, n=30, fileids=subset)
    # plot_term_coocurrance_matrix(corpus, n=30, fileids=subset)
    # plot_term_occurance_over_time(corpus, n=30, fileids=subset)
    # plot_tsne_clusters(corpus, fileids=subset)

    # # with clusters
    # model = Pipeline([
    #     ("norm", Corpus_Vectorizer.TitleNormalizer()),
    #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
    #     ('clusters', Corpus_Cluster.MiniBatchKMeansClusters(k=3))
    # ])
    #
    # docs = corpus.title_tagged(fileids=subset)
    # clusters = model.fit_transform(docs)
    #
    # plot_tsne_clusters(corpus, fileids=subset, labels=clusters)

