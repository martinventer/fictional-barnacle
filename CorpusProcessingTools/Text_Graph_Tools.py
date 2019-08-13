#! envs/fictional-barnacle/bin/python3.6
"""
Text_Graph_tools.py

@author: martinventer
@date: 2019-08-13

Graph tools for text data
"""



if __name__ == "__main__":
    from CorpusReaders import Elsevier_Corpus_Reader
    # from CorpusProcessingTools import Corpus_Vectorizer, Corpus_Cluster
    # from sklearn.pipeline import Pipeline

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)
    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 10, shuffle=False)
    subset_fileids = next(loader.fileids(test=True))

