#! envs/fictional-barnacle/bin/python3.6
"""
Corpus_Vectorizer.py

@author: martinventer
@date: 2019-06-28

Tools Clustering corpus
"""

from sklearn.pipeline import Pipeline

import TextTools.Transformers
from TextTools.Transformers import KMeansClusters, HierarchicalClustering

if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader

    from time import time

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)
    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 10, shuffle=False)
    subset_fileids = next(loader.fileids(test=True))

    # --------------------------------------------------------------------------
    # KMeansClusters
    # --------------------------------------------------------------------------
    docs = list(corpus.title_tagged(fileids=subset_fileids))
    observations = list(corpus.title_tagged(fileids=subset_fileids))

    # Text2FrequencyVector
    prepare_data = Pipeline([('normalize', TextTools.Transformers.TextNormalizer()),
                             ('vectorize',
                              TextTools.Transformers.Text2FrequencyVector())])
    X = prepare_data.fit_transform(observations)

    model = KMeansClusters(k=10)
    labels = model.fit_transform(X)

    Cluster_Plotting.plot_clusters_2d(X, labels)




