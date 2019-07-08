#! envs/fictional-barnacle/bin/python3.6
"""
Context_Extraction.py

@author: martinventer
@date: 2019-07-05

Tools for feature extraction using context aware features
"""

from nltk import ne_chunk
from itertools import groupby
from nltk.corpus import wordnet as wn
from nltk.chunk import tree2conlltags
from nltk.probability import FreqDist
from nltk.chunk.regexp import RegexpParser
from unicodedata import category as unicat
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.collocations import QuadgramCollocationFinder
from nltk.metrics.association import QuadgramAssocMeasures

from nltk.collocations import TrigramCollocationFinder
from nltk.metrics.association import TrigramAssocMeasures

from nltk.collocations import BigramCollocationFinder
from nltk.metrics.association import BigramAssocMeasures


GRAMMAR = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
GOODTAGS = frozenset(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])
GOODLABELS = frozenset(['PERSON', 'ORGANIZATION', 'FACILITY', 'GPE', 'GSP'])


class KeyphraseExtractor(BaseEstimator, TransformerMixin):
    """
    Wraps a PickledCorpusReader consisting of pos-tagged documents.
    """
    def __init__(self, grammar=GRAMMAR):
        self.grammar = GRAMMAR
        self.chunker = RegexpParser(self.grammar)

    def normalize(self, sent):
        """
        Removes punctuation from a tokenized/tagged sentence and
        lowercases words.
        """
        is_punct = lambda word: all(unicat(char).startswith('P') for char in word)
        sent = filter(lambda t: not is_punct(t[0]), sent)
        sent = map(lambda t: (t[0].lower(), t[1]), sent)
        return list(sent)

    def extract_keyphrases(self, document):
        """
        For a document, parse sentences using our chunker created by
        our grammar, converting the parse tree into a tagged sequence.
        Yields extracted phrases.
        """
        # for sents in document: removed so that it can deal with titles
        for sent in document:
            # for sent in sents:
                sent = self.normalize(sent)
                if not sent: continue
                chunks = tree2conlltags(self.chunker.parse(sent))
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

    def transform(self, documents):
        for document in documents:
            yield list(self.extract_keyphrases(document))


class EntityExtractor(BaseEstimator, TransformerMixin):
    """
    Extract entities from a sentance, such as Keyphrases
    """
    def __init__(self, labels=GOODLABELS, **kwargs):
        self.labels = labels

    def get_entities(self, document):
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
        # print('transform')
        for document in documents:
            yield self.get_entities(document)


class SignificantCollocations(BaseEstimator, TransformerMixin):
    """
    Find and rank significant collocations in text.
    """
    def __init__(self,
                 ngram_class=QuadgramCollocationFinder,
                 metric=QuadgramAssocMeasures.pmi):
        self.ngram_class = ngram_class
        self.metric = metric

    def fit(self, docs, target=None):
        ngrams = self.ngram_class.from_documents(docs)
        self.scored_ = dict(ngrams.score_ngrams(self.metric))

    # def transform(self, docs):
    #     for doc in docs:
    #         ngrams = self.ngram_class.from_words(doc)
    #         yield {
    #             ngram: self.scored_.get(ngrams, 0.0)
    #             for ngram in ngrams.nbest(QuadgramAssocMeasures.raw_freq, 50)
    #         }
    def transform(self, docs):
        ngrams = self.ngram_class.from_words(docs)
        yield {
            ngram: self.scored_.get(ngrams, 0.0)
            for ngram in ngrams.nbest(QuadgramAssocMeasures.raw_freq, 50)
        }


def rank_quadgrams(docs, metric, path=None):
    """
    Find and rank quadgrams from the supplied corpus using the given
    association metric. Write the quadgrams out to the given path if
    supplied otherwise return the list in memory.
    """

    # Create a collocation ranking utility from corpus words.
    ngrams = QuadgramCollocationFinder.from_words(docs)

    # Rank collocations by an association metric
    scored = ngrams.score_ngrams(metric)

    if path:
        with open(path, 'w') as f:
            f.write("Collocation\tScore ({})\n".format(metric.__name__))
            for ngram, score in scored:
                f.write("{}\t{}\n".format(repr(ngram), score))
    else:
        return scored


class RankGrams(BaseEstimator, TransformerMixin):
    """
    Constructs and ranks ngrams for a given corpus
    """
    def __init__(self, n=2, path=None):
        self.n = n
        self.path = path
        if self.n == 2:
            self.metric = BigramAssocMeasures.likelihood_ratio
        elif self.n == 3:
            self.metric = TrigramAssocMeasures.likelihood_ratio
        elif self.n == 4:
            self.metric = QuadgramAssocMeasures.likelihood_ratio
        else:
            print("error, order of n-gram not supported")

    def rank_grams(self, docs):
        """
        Find and rank gram from the supplied corpus using the given
        association metric. Write the quadgrams out to the given path if
        supplied otherwise return the list in memory.
        """
        # Create a collocation ranking utility from corpus words.
        if self.n == 2:
            self.ngrams = BigramCollocationFinder.from_words(docs)
        elif self.n == 3:
            self.ngrams = TrigramCollocationFinder.from_words(docs)
        elif self.n == 4:
            self.ngrams = QuadgramCollocationFinder.from_words(docs)

        # Rank collocations by an association metric
        self.scored = self.ngrams.score_ngrams(self.metric)

    def fit(self, docs=None, **kwargs):
        return self

    def transform(self, docs, **kwargs):
        self.rank_grams(docs, **kwargs)

        if self.path:
            with open(self.path, 'w') as f:
                f.write("Collocation\tScore ({})\n".format(
                    self.metric.__name__))
                for ngram, score in self.scored:
                    f.write("{}\t{}\n".format(repr(ngram), score))
        else:
            return self.scored



if __name__ == '__main__':
    from CorpusReader import Elsevier_Corpus_Reader
    from CorpusProcessingTools import Corpus_Vectorizer

    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        "Corpus/Processed_corpus/")

    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 12, shuffle=False)
    subset = next(loader.fileids(test=True))

    docs = list(corpus.title_tagged(fileids=subset))

    # # Key Phrase extractor
    # phrase_extractor = KeyphraseExtractor()
    # keyphrases = list(phrase_extractor.fit_transform(docs))
    # print(keyphrases[0])

    # # Entity extractor
    # entity_extractor = EntityExtractor()
    # entities = list(entity_extractor.fit_transform(docs))
    # print(entities[0])

    # # significant collocations
    # docs = list(corpus.title_words(fileids=subset))
    # sig = SignificantCollocations()
    # sig.fit(docs)
    # ents = sig.transform(docs)

    # ranked collocations
    docs = corpus.title_words(fileids=subset)

    # dat = rank_quadgrams(
    #     docs, QuadgramAssocMeasures.likelihood_ratio
    # )

    rg = RankGrams(n=4)

    dat = rg.transform(
        docs
    )

    for gram, score in dat:
        if "soft" in gram and score > 2500:
            print(gram, score)

    for i in range(20):
        print(dat[i])
