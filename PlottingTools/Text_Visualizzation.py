#! envs/fictional-barnacle/bin/python3.6
"""
Text_Visualization.py

@author: martinventer
@date: 2019-07-08

Tools for Visualizing Text related data
"""

import matplotlib.pyplot as plt

if __name__ == '__main__':
    from CorpusReader import Elsevier_Corpus_Reader
    from CorpusProcessingTools import Corpus_Vectorizer

    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 12, shuffle=False)
    subset = next(loader.fileids(test=True))

    docs = list(corpus.title_tagged(fileids=subset))

    # simple token frequency over time plot
    fig, ax = plt.subplots(figsize=(9, 6))

    for term in terms:
        data[term].plot(ax=ax)

    ax.set_title("Token frequency, over time")
    ax.set_ylabel("word count")
    ax.set_xlabel("publication date")
    ax.legend()
    plt.show
