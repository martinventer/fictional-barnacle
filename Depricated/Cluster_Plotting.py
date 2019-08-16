# #! envs/fictional-barnacle/bin/python3.6
# """
# Cluster_Plotting.py
#
# @author: martinventer
# @date: 2019-07-03
#
# Tools for plotting clusters
# """
#
#
# from sklearn.pipeline import Pipeline
#
# from TextTools.Analysis import plot_clusters
#
# if __name__ == '__main__':
#     from CorpusReaders import Elsevier_Corpus_Reader
#     from Depricated import Corpus_Cluster, Corpus_Vectorizer
#
#     corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
#         "Corpus/Processed_corpus/")
#
#     loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 100, shuffle=False)
#     subset = next(loader.fileids(test=True))
#
#     docs = list(corpus.title_tagged(fileids=subset))
#
#     # --------------------------------------------------------------------------
#     # plot_dendrogram
#     # --------------------------------------------------------------------------
#     # model = Pipeline([
#     #     ("norm", Corpus_Vectorizer.TitleNormalizer()),
#     #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
#     #     ('clusters', Corpus_Cluster.HierarchicalClustering())
#     # ])
#     #
#     # clusters = model.fit_transform(observations)
#     # labels = model.named_steps['clusters'].labels
#     # children = model.named_steps['clusters'].children
#     #
#     # plot_dendrogram(children)
#
#     # --------------------------------------------------------------------------
#     # plot_clusters
#     # --------------------------------------------------------------------------
#     # # decompose data to 2D
#     # reduce = Pipeline([
#     #     ("norm", Corpus_Vectorizer.TitleNormalizer()),
#     #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
#     #     ('pca', PCA(n_components=2))
#     # ])
#     #
#     # X2d = reduce.fit_transform(docs)
#     #
#     # # plot Kmeans
#     # model = Pipeline([
#     #     ("norm", Corpus_Vectorizer.TitleNormalizer()),
#     #     ("vect", Corpus_Vectorizer.OneHotVectorizer()),
#     #     ('clusters', Corpus_Cluster.MiniBatchKMeansClusters(k=3))
#     # ])
#     #
#     # clusters = model.fit_transform(docs)
#     #
#     # plot_clusters(X2d, clusters)
#
#
