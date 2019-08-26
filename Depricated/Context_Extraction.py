# #! envs/fictional-barnacle/bin/python3.6
# """
# Context_Extraction.py
#
# @author: martinventer
# @date: 2019-07-05
#
# Tools for feature extraction using context aware features
# """
#
# from TextTools.Transformers import KeyphraseExtractorS
#
# GOODTAGS = frozenset(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])
#
# #
# #
# # class SignificantCollocations(BaseEstimator, TransformerMixin):
# #     """
# #     Find and rank significant collocations in text.
# #     """
# #     def __init__(self,
# #                  ngram_class=QuadgramCollocationFinder,
# #                  metric=QuadgramAssocMeasures.pmi):
# #         """
# #         construct the collocations
# #         Parameters
# #         ----------
# #         ngram_class :
# #         metric
# #         """
# #         self.ngram_class = ngram_class
# #         self.metric = metric
# #         self.scored_ = None
# #         self.tokenizer = Corpus_Vectorizer.TextSimpleTokenizer()
# #
# #     def fit(self, observations, target=None):
# #         print("fit")
# #         observations = self.tokenizer.fit_transform(observations)
# #         ngrams = self.ngram_class.from_documents(observations)
# #         self.scored_ = dict(ngrams.score_ngrams(self.metric))
# #
# #     def transforman(self, observations):
# #         print('transform')
# #         for doc in observations:
# #             # print(doc)
# #             # doc = self.tokenizer.fit_transform(doc)
# #             print(doc)
# #             ngrams = self.ngram_class.from_words(observations)
# #             # print(ngrams)
# #             yield {
# #                 ngram: self.scored_.get(ngrams, 0.0)
# #                 for ngram in ngrams.nbest(QuadgramAssocMeasures.raw_freq, 10)
# #             }
#
#
# if __name__ == '__main__':
#     from CorpusReaders import Elsevier_Corpus_Reader
#
#     root = "Tests/Test_Corpus/Processed_corpus/"
#     corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
#         root=root)
#     loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 10, shuffle=False)
#     subset_fileids = next(loader.fileids(test=True))
#
#     # --------------------------------------------------------------------------
#     # KeyphraseExtractorL
#     # --------------------------------------------------------------------------
#     # observations = corpus.title_tagged(fileids=subset_fileids)
#     #
#     # phrase_extractor = KeyphraseExtractorL()
#     # keyphrases = list(phrase_extractor.fit_transform(observations))
#     # print(keyphrases[:4])
#
#     # --------------------------------------------------------------------------
#     # KeyphraseExtractorS
#     # --------------------------------------------------------------------------
#     docs = corpus.title_tagged(fileids=subset_fileids)
#
#     phrase_extractor = KeyphraseExtractorS()
#     keyphrases = list(phrase_extractor.fit_transform(docs))
#     print(keyphrases[:20])
#
#     # --------------------------------------------------------------------------
#     # EntityExtractor
#     # --------------------------------------------------------------------------
#     # observations = corpus.title_tagged(fileids=subset_fileids)
#     #
#     # entity_extractor = EntityExtractor()
#     # entities = list(entity_extractor.fit_transform(observations))
#     # print(entities[:4])
#
#     # --------------------------------------------------------------------------
#     # RankGrams
#     # --------------------------------------------------------------------------
#     # observations = corpus.title_word(fileids=subset_fileids)
#     #
#     # ranker = RankGrams(n_terms=4)
#     # ranks = ranker.fit_transform(observations)
#     # print(ranks[:4])
#
