#! envs/fictional-barnacle/bin/python3.6
"""
Context_Extraction.py

@author: martinventer
@date: 2019-07-05

Tools for feature extraction using context aware features
"""

from nltk import ne_chunk
from nltk.corpus import wordnet as wn
from nltk.chunk import tree2conlltags
from nltk.probability import FreqDist
from nltk.chunk.regexp import RegexpParser
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import QuadgramCollocationFinder, \
    TrigramCollocationFinder, BigramCollocationFinder
from nltk.metrics.association import QuadgramAssocMeasures, \
    TrigramAssocMeasures, BigramAssocMeasures

from unicodedata import category as unicat
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import groupby

from Utils.Utils import is_punct
from CorpusProcessingTools import Corpus_Vectorizer

GOODTAGS = frozenset(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])


class KeyphraseExtractorL(BaseEstimator, TransformerMixin):
    """
    Transformer for pos-tagged documents that outputs keyphrases. Converts a
    corpus into a bag of key phrases.
    """
    GRAMMAR = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'

    def __init__(self, grammar=None):
        """
        sets up he keyphrase extractor to have a grammer and parser
        Parameters
        ----------
        grammar
        """
        if not grammar:
            self.grammar = KeyphraseExtractorL.GRAMMAR
        else:
            self.grammar = grammar
        self.chunker = RegexpParser(self.grammar)

    def normalize(self, sent):
        """
        Removes punctuation from a tokenized/tagged sentence and
        lowercases words.
        """
        sent = filter(lambda t: not is_punct(t[0]), sent)
        sent = map(lambda t: (t[0].lower(), t[1]), sent)
        return list(sent)

    def extract_keyphrases(self, document):
        """
        For a document, parse sentences using the chunker created by
        our grammar, converting the parse tree into a tagged sequence.
        Yields extracted phrases.
        """
        for sent in document:
            # remove punctuation and make lowercase
            sent = self.normalize(sent)
            # check that the sentence is not empty now.
            if not sent:
                continue
            # the output of the chunk parser is a tree, which we convert to
            # and IOM-tagged format, IOB tells you what a word does in the
            # sentence
            chunks = tree2conlltags(self.chunker.parse(sent))
            # group the chunks, if the group is key, process its contents
            phrases = [
                " ".join(word for word, pos, chunk in group).lower()
                for key, group in groupby(
                    chunks, lambda term: term[-1] != 'O'
                ) if key
            ]
            for phrase in phrases:
                yield phrase

    def fit(self, documents, y=None):
        return self

    def transform(self, documents) -> list:
        for document in documents:
            yield list(self.extract_keyphrases(document))


class KeyphraseExtractorS(KeyphraseExtractorL):
    """
    Transformer for pos-tagged documents that outputs keyphrases. Converts a
    corpus into a bag of key phrases. one list per doc
    """
    def __init__(self):
        """
        sets up he keyphrase extractor to have a grammer and parser
        Parameters
        ----------
        """
        KeyphraseExtractorL.__init__(self)

    def process_keyphrase(self, document) -> list:
        """
        removes single word keyphrases and connects remaining phrases with "-"
        Parameters
        ----------
        phrase: str
            singel keyphrase
        Returns
        -------
            str
        """
        phrase_list = list()
        for phrase in self.extract_keyphrases(document):
            phrase = phrase.replace(" ", "-")
            if "-" in phrase:
                phrase_list.append(phrase)

        return phrase_list

    def transform(self, documents) -> list:
        for document in documents:
            yield self.process_keyphrase(document)


class EntityExtractor(BaseEstimator, TransformerMixin):
    """
    Extract entities from a sentance,Converts a corpus into a bag of entities.
    """
    GOODLABELS = frozenset(['PERSON', 'ORGANIZATION', 'FACILITY', 'GPE', 'GSP'])

    def __init__(self, labels=None, **kwargs):
        """
        setup an entity extractor
        Parameters
        ----------
        labels: a set of lable catagories to be extracted.
        """
        # Add the default labels if not passed into the class.
        if not labels:
            self.labels = EntityExtractor.GOODLABELS
        else:
            self.labels = labels

    def get_entities(self, document) -> list:
        """
        Converts each sentence in a document into a chunked tree structure
        Parameters
        ----------
        document

        Returns
        -------

        """
        entities = []
        # for paragraph in document:  # updated for title data
        for sentence in document:
            # for sentence in paragraph:  # removed to function for title data
                trees = ne_chunk(sentence)
                for tree in trees:
                    if hasattr(tree, 'label'):
                        if tree.label() in self.labels:
                            entities.append(
                                ' '.join([child[0].lower() for child in tree])
                                )
        return entities

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        return [self.get_entities(document) for document in documents]
        # for document in documents:
        #     yield self.get_entities(document)
#
#
# class SignificantCollocations(BaseEstimator, TransformerMixin):
#     """
#     Find and rank significant collocations in text.
#     """
#     def __init__(self,
#                  ngram_class=QuadgramCollocationFinder,
#                  metric=QuadgramAssocMeasures.pmi):
#         """
#         construct the collocations
#         Parameters
#         ----------
#         ngram_class :
#         metric
#         """
#         self.ngram_class = ngram_class
#         self.metric = metric
#         self.scored_ = None
#         self.tokenizer = Corpus_Vectorizer.TextSimpleTokenizer()
#
#     def fit(self, observations, target=None):
#         print("fit")
#         observations = self.tokenizer.fit_transform(observations)
#         ngrams = self.ngram_class.from_documents(observations)
#         self.scored_ = dict(ngrams.score_ngrams(self.metric))
#
#     def transforman(self, observations):
#         print('transform')
#         for doc in observations:
#             # print(doc)
#             # doc = self.tokenizer.fit_transform(doc)
#             print(doc)
#             ngrams = self.ngram_class.from_words(observations)
#             # print(ngrams)
#             yield {
#                 ngram: self.scored_.get(ngrams, 0.0)
#                 for ngram in ngrams.nbest(QuadgramAssocMeasures.raw_freq, 10)
#             }


class RankGrams(BaseEstimator, TransformerMixin):
    """
    Constructs and ranks ngrams for a given corpus
    """
    def __init__(self, n=2):
        self.n = n
        if self.n == 2:
            self.metric = BigramAssocMeasures.likelihood_ratio
            self.model = BigramCollocationFinder
        elif self.n == 3:
            self.metric = TrigramAssocMeasures.likelihood_ratio
            self.model = TrigramCollocationFinder
        elif self.n == 4:
            self.metric = QuadgramAssocMeasures.likelihood_ratio
            self.model = QuadgramCollocationFinder
        else:
            print("error, order of n_terms-gram not supported")

        self.grams = None
        self.scored = None

    def rank_grams(self, docs, **kwargs):
        """
        Find and rank gram from the supplied corpus using the given
        association metric. Write the quadgrams out to the given path if
        supplied otherwise return the list in memory.
        """
        # Create a collocation ranking utility from corpus words.
        self.grams = self.model.from_words(docs)

        # Rank collocations by an association metric
        return self.grams.score_ngrams(self.metric)

    def fit(self, docs=None, **kwargs):
        return self

    def transform(self, docs, **kwargs):
        return self.rank_grams(docs, **kwargs)


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)
    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 10, shuffle=False)
    subset_fileids = next(loader.fileids(test=True))

    # --------------------------------------------------------------------------
    # KeyphraseExtractorL
    # --------------------------------------------------------------------------
    # observations = corpus.title_tagged(fileids=subset_fileids)
    #
    # phrase_extractor = KeyphraseExtractorL()
    # keyphrases = list(phrase_extractor.fit_transform(observations))
    # print(keyphrases[:4])

    # --------------------------------------------------------------------------
    # KeyphraseExtractorS
    # --------------------------------------------------------------------------
    docs = corpus.title_tagged(fileids=subset_fileids)

    phrase_extractor = KeyphraseExtractorS()
    keyphrases = list(phrase_extractor.fit_transform(docs))
    print(keyphrases[:20])

    # --------------------------------------------------------------------------
    # EntityExtractor
    # --------------------------------------------------------------------------
    # observations = corpus.title_tagged(fileids=subset_fileids)
    #
    # entity_extractor = EntityExtractor()
    # entities = list(entity_extractor.fit_transform(observations))
    # print(entities[:4])

    # --------------------------------------------------------------------------
    # RankGrams
    # --------------------------------------------------------------------------
    # observations = corpus.title_word(fileids=subset_fileids)
    #
    # ranker = RankGrams(n_terms=4)
    # ranks = ranker.fit_transform(observations)
    # print(ranks[:4])

