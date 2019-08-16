#! envs/fictional-barnacle/bin/python3.6
"""
TextTools.py

@author: martinventer
@date: 2019-08-14

Tools for processing text fields extracted from a corpus
"""
import string

import nltk
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk import WordNetLemmatizer
from nltk.cluster import KMeansClusterer
from nltk.corpus import wordnet as wn
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from Utils.Utils import is_punct, identity


class TextStemTokenize(BaseEstimator, TransformerMixin):
    """
    Stems the tokens in a list of paragraphs list of sentances list of
    token, tag tuples, returning a list of stemmed words for each document
    """
    def __init__(self, language='english'):
        self.stemmer = nltk.stem.SnowballStemmer(language)

    def stem(self, doc) -> list:
        """
        takes a word makes it lower case, and returns the stemmed from.
        Parameters
        ----------
        doc : list
            list of paragraphs list of sentances list of token, tag tuples

        Returns
        -------
            a single list of stemmed words
        """
        word_list = []
        for sent in doc:
            for token, tag in sent:
                if token in string.punctuation:
                    continue
                word_list.append(self.stemmer.stem(token).lower())
        return word_list

    def fit(self, documents):
        return self

    def transform(self, documents):
        return [
            self.stem(document)
            for document in documents
        ]


class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    transformer that normalizes and lemmitizes text. THis transformer needs
    text in the form that one document is a list of sentences, which is a list
    of (token, tag) tuples.
    """
    def __init__(self, language='english'):
        """
        Initialize the text normalizer with the wordnet lemmatizer and the nltk
        stopwords in a selected language.
        Parameters
        ----------
        language : str
            string indication of the language
        """
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_stopword(self, token) -> bool:
        """
        returns a boolean True if a string is in the list of stopwords
        Parameters
        ----------
        token : str
            string token

        Returns
        -------
            Bool

        """
        return token.lower() in self.stopwords

    def lemmatize(self, token, pos_tag) -> str:
        """
        returns a lametized string
        Parameters
        ----------
        pos_tag : str
            string symobol containing the part of speach tag for a given word
        token : str
            string token

        Returns
        -------
            lematized word

        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def normalize(self, document) -> list:
        """
        returns a list of normalized words
        Parameters
        ----------
        document : list
            list of paragraphs that are lists of sentences that are lists of
            token tag tuples.

        Returns
        -------
            list of normalized words

        """
        return [
            self.lemmatize(token, tag).lower()
            for sentence in document
            for (token, tag) in sentence
            if not is_punct(token) and not self.is_stopword(token)
        ]

    def fit(self, documents):
        return self

    def transform(self, documents) -> list:
        """
        yields a list of normalized text for each document passed
        Parameters
        ----------
        documents: list of documents, or generator returning documents as as
        list of paragraps list of sents as (token, tag)

        Returns
        -------
            list of lametized lowercase text with punctuation removed
        """
        return [
            self.normalize(document)
            for document in documents
        ]


class CorpusSimpleNormalizer(BaseEstimator, TransformerMixin):
    """
    wrapper a  [[(token, tag]] corpus item that will return only lemmatized,
    lowercase words without punctuation
    """

    def __init__(self, language='english'):
        """
        Initialize the text normalizer with the wordnet lemmatizer and the nltk
        stopwords in a selected language.
        Parameters
        ----------
        language : str
            string indication of the language
        """
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_stopword(self, token) -> bool:
        """
        returns a boolean True if a string is in the list of stopwords
        Parameters
        ----------
        token : str
            string token

        Returns
        -------
            Bool

        """
        return token.lower() in self.stopwords

    def lemmatize(self, token, pos_tag) -> str:
        """
        returns a lametized string
        Parameters
        ----------
        pos_tag : str
            string symobol containing the part of speach tag for a given word
        token : str
            string token

        Returns
        -------
            lematized word

        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def normalize(self, document) -> list:
        """
        returns a list of normalized words
        Parameters
        ----------
        document : list
            list of paragraphs that are lists of sentences that are lists of
            token tag tuples.

        Returns
        -------
            list of normalized words

        """
        return [
            self.lemmatize(token, tag).lower()
            for sentence in document
            for (token, tag) in sentence
            if not is_punct(token) and not self.is_stopword(token)
        ]

    def fit(self, documents):
        return self

    def transform(self, documents):
        return [
            [term for term in self.normalize(document)]
            for document in documents]


class TextSimpleTokenizer(BaseEstimator, TransformerMixin):
    """
    transformer that takes text in the form of list of sentances that are
    lists of token, tag tuples and returns a single list of strings for each
    document
    """
    def __init__(self):
        """

        """
        pass

    def get_words(self, document):
        return [
            token
            for sentance in document
            for (token, tag) in sentance
        ]

    def fit(self, documents):
        return self

    def transform(self, documents):
        return [
            self.get_words(document)
            for document in documents]


class Text2wordCountVector(CountVectorizer):
    """
    Wrapper for CountVectoriser that converts a corpus into a matrix of term
    occurrences per document. Requires the input data to be in the form of
    one list of words per document. does not make use of a preprocessor of
    force lowercase
    """
    def __init__(self):
        CountVectorizer.__init__(self, tokenizer=identity,
                                 preprocessor=None,
                                 lowercase=False)


class Text2FrequencyVector(CountVectorizer):
    """
    Wrapper for CountVectoriser that converts a corpus into a matrix of term
    occurrences per document. Requires the input data to be in the form of
    one list of words per document.
    """
    def __init__(self):
        CountVectorizer.__init__(self, tokenizer=identity,
                                 preprocessor=None,
                                 lowercase=False)


class Text2OneHotVector(CountVectorizer):
    """
    Wrapper for CountVectoriser that converts a corpus into a matrix of one
    hot vectors per document. Requires the input data to be in the form of
    one list of words per document. This can be done using
    """
    def __init__(self):
        CountVectorizer.__init__(self, binary=True,
                                 tokenizer=identity,
                                 preprocessor=None,
                                 lowercase=False)


class Text2TFIDVector(TfidfVectorizer):
    """
    Wrapper for TfidfVectorizer that converts a corpus into a matrix of
    tifidf vectors per document. Requires the input data to be in the form of
    one list of words per document. This can be done using    """
    def __init__(self):
        TfidfVectorizer.__init__(self, tokenizer=identity,
                                 preprocessor=None,
                                 lowercase=False)


class Text2Doc2VecVector(BaseEstimator, TransformerMixin):
    """
    transformer that converts a corpus into a matrix of
    doc2vec vectors per document. Requires the input data to be in the form of
    one list of words per document. This can be done using one of the text
    normalizers
    """
    def __init__(self, vector_size=50, min_count=0):
        """
        Parameters
        ----------
        vector_size : int
            Desired vector length for output
        min_count : int
            Doc2Vec will ignore any tokens with a count below this number

        """
        self.min_count = min_count
        self.vector_size = vector_size

    @staticmethod
    def gensim_docs(documents) -> list:
        """
        Convert the raw input observations to a tagged document list
        Parameters
        ----------
        documents : list

        Returns
        -------
            tagged document list
        """
        docs = [
            TaggedDocument(words, ['d{}'.format(idx)])
            for idx, words in enumerate(documents)
        ]
        return docs

    def fit(self, documents):
        docs = self.gensim_docs(documents)
        model = Doc2Vec(docs,
                        vector_size=self.vector_size,
                        min_count=self.min_count)
        self.model = model
        return self

    def transform(self, documents):
        temp = sparse.csr_matrix(self.model.docvecs.doctag_syn0)
        return temp


class KMeansClusters(BaseEstimator, TransformerMixin):
    """
    Cluster text data using k-means. Makes use of nltk k-means clustering.
    Allows for alternative distance measures
    """

    def __init__(self, k=7, distance=None):
        """
        initializes the kmeans clustering
        Parameters
        ----------
        k : int
            number of clusters desired
        distance :
            an nltk.cluster.util for an alternative distance measure
        """
        self.k = k
        if not distance:
            self.distance = nltk.cluster.util.cosine_distance
        else:
            self.distance = distance
        self.model = KMeansClusterer(self.k, self.distance,
                                     avoid_empty_clusters=True)

    def fit(self, text_vector):
        return self

    def transform(self, text_vector):
        """
        fits the K-means model to the given documents
        Parameters
        ----------
        text_vector :
            a matrix of vectorised documents, each row contains a vector for
            each document

        Returns
        -------
            ndarray of document labels
        """
        if type(text_vector) is sparse.csr.csr_matrix:
            text_vector = text_vector.toarray()
        return np.array(self.model.cluster(text_vector, assign_clusters=True))


class HierarchicalClustering(BaseEstimator, TransformerMixin):
    """
    wrapper for the
    """

    def __init__(self, **kwargs):
        self.model = AgglomerativeClustering(**kwargs)
        self.children = None

    def fit(self, text_vector):
        return self

    def transform(self, text_vector):
        """
        fits an agglomerative clustering to given vector
        Parameters
        ----------
        text_vector :
            a matrix of vectorised documents, each row contains a vector for
            each document

        Returns
        -------
            ndarray of document labels

        """
        if type(text_vector) is sparse.csr.csr_matrix:
            text_vector = text_vector.toarray()

        out = self.model.fit_predict(text_vector)
        self.children = self.model.children_

        return np.array(out)


class SklearnTopicModels(BaseEstimator, TransformerMixin):
    """
    a topic modeler that identifies the main topics in a corpus of documents
    """

    def __init__(self, n_topics=50, estimator='LDA', vectorizor="tfidvec"):
        """
        a topic modler calling form sklearn decomposition libray
        Parameters
        ----------
        n_topics : int
            number of topics assigned to the
        estimator : str
            'LDA' Latent Dirichlet Allocation (default)
            'LSA' Latent Semantic Analysis
            'NMF' Non-Negative Matrix Factorization
        vectorizor : str
            'freqvec' term frequency vector
            'tfidvec' term frequency-inverse document frequency vector
            'onehotvec' term frequency vector with one hot encoding
        """
        self.n_topics = n_topics

        if estimator is not "LDA":
            if estimator == 'LSA':
                self.estimator = TruncatedSVD(n_components=self.n_topics)
            elif estimator == 'NMF':
                self.estimator = NMF(n_components=self.n_topics)
        else:
            self.estimator = LatentDirichletAllocation(
                n_components=self.n_topics)

        if vectorizor is not 'tfidvec':
            if vectorizor is 'freqvec':
                self.vectorizor = Text2FrequencyVector()
            elif vectorizor is 'onehotvec':
                self.vectorizor = Text2OneHotVector()

        else:
            self.vectorizor = Text2TFIDVector()

        self.model = Pipeline([
            ('norm', TextNormalizer()),
            ('vect', self.vectorizor),
            ('model', self.estimator)
        ])

    def get_topics(self, n_term=25):
        """
        n_terms is the number of top terms to show for each topic
        """
        vectorizer = self.model.named_steps['vect']
        model = self.model.steps[-1][1]
        names = vectorizer.get_feature_names()
        topics = dict()

        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[:-(n_term - 1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens

        return topics

    def fit(self, documents):
        return self

    def transform(self, documents):
        self.model.fit_transform(documents)

        return self.model


if __name__ == '__main__':
    from CorpusReaders import Elsevier_Corpus_Reader

    root = "Tests/Test_Corpus/Processed_corpus/"
    corpus = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
        root=root)
    loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(corpus, 10, shuffle=False)
    subset_fileids = next(loader.fileids(test=True))

    # ==========================================================================
    # TextNormalizer
    # ==========================================================================
    # normal = TextNormalizer()
    #
    # input = ['.', ',', 'i', '?', 't.', '.t']
    # for word in input:
    #     print(word, is_punct(word))
    #
    # input = ['.', ',', 'i', '?', 't.', '.t', 'the', 'Steven']
    # for word in input:
    #     print(word, normal.is_stopword(word))
    #
    # input = [('gardening', 'V')]
    # for token, tag in input:
    #     print(token, tag, normal.lemmatize(token, tag))
    #
    # input = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
    #           ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
    #           ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
    #           ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
    #           ('sensors', 'NNS')]]]
    # for doc in input:
    #     print(normal.normalize(doc))
    #
    # input = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
    #            ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
    #            ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
    #            ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
    #            ('sensors', 'NNS')]]]
    # print(normal.fit_transform(input))
    #
    # aa = normal.transform(corpus.title_tagged())

    # ==========================================================================
    # TextStemTokenize
    # ==========================================================================
    # stemmed = TextStemTokenize()
    #
    # input = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
    #           ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
    #           ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
    #           ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
    #           ('sensors', 'NNS')]]]
    # for doc in input:
    #     print(stemmed.stem(doc))

    # ==========================================================================
    # TextSimpleTokenizer
    # ==========================================================================
    # simple = TextSimpleTokenizer()
    #
    # input = [[[('A', 'DT'), ('study', 'NN'), ('of', 'IN'), ('laundry', 'JJ'),
    #           ('tidiness', 'NN'), (':', ':'), ('Laundry', 'JJ'),
    #           ('state', 'NN'), ('determination', 'NN'), ('using', 'VBG'),
    #           ('video', 'NN'), ('and', 'CC'), ('3D', 'CD'),
    #           ('sensors', 'NNS')]]]
    # for doc in input:
    #     print(simple.get_words(doc))

    # ==========================================================================
    # Text2FrequencyVector
    # ==========================================================================
    # simple = TextSimpleTokenizer()
    # vec = Text2FrequencyVector()
    #
    # input_text = [['From', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
    #               'realization','of', 'qualitative', 'artificial',
    #                'intelligence', 'based', 'on', 'task','realizability',
    #                'analysis']]
    # vector = vec.fit_transform(input_text)
    # for row in vector:
    #     print(row.sum())
    #
    # input_text = simple.transform(corpus.title_tagged())
    # vector = vec.fit_transform(input_text)
    #
    # print("{} documents, {} words".format(vector.shape[0], vector.shape[1]))
    # print(type(vector))

    # ==========================================================================
    # Text2OneHotVector
    # ==========================================================================
    # simple = TextSimpleTokenizer()
    # vec = Text2OneHotVector()
    #
    # input_text = [['From', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
    #                'realization', 'of', 'qualitative', 'artificial',
    #                'intelligence', 'based', 'on', 'task', 'realizability',
    #                'analysis', 'analysis']]
    # vector = vec.fit_transform(input_text)
    # for row in vector:
    #     print(row.sum())
    #
    # input_text = simple.transform(corpus.title_tagged())
    # vector = vec.fit_transform(input_text)
    #
    # print("{} documents, {} words".format(vector.shape[0], vector.shape[1]))
    # print(type(vector))

    # ==========================================================================
    # Text2TFIDVector
    # ==========================================================================
    # simple = TextSimpleTokenizer()
    # vec = Text2TFIDVector()
    #
    # input_text = [['From', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
    #                'realization', 'of', 'qualitative', 'artificial',
    #                'intelligence', 'based', 'on', 'task', 'realizability',
    #                'analysis', 'analysis']]
    # vector = vec.fit_transform(input_text)
    # for row in vector:
    #     print(row)
    #
    # input_text = simple.transform(corpus.title_tagged())
    # vector = vec.fit_transform(input_text)
    #
    # print("{} documents, {} words".format(vector.shape[0], vector.shape[1]))
    # print(type(vector))

    # ==========================================================================
    # Text2Doc2VecVector
    # ==========================================================================
    # simple = TextSimpleTokenizer()
    # vec = Text2Doc2VecVector(vector_size=3,
    #                           min_count=0)
    #
    # input_text = [['From', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
    #                'realization', 'of', 'qualitative', 'artificial',
    #                'intelligence', 'based', 'on', 'task', 'realizability',
    #                'analysis', 'analysis'],
    #               ['try', 'AlphaGo', 'to', 'BetaGo', '-', 'Quantitative',
    #                'realization', 'of', 'qualitative', 'artificial',
    #                'intelligence', 'based', 'on', 'task', 'realizability',
    #                'analysis']
    #               ]
    # vector = vec.fit_transform(input_text)
    # for row in vector:
    #     print(row)
    #
    # input_text = simple.transform(corpus.title_tagged())
    # vector = vec.fit_transform(input_text)
    #
    # print("{} documents, {} components".format(vector.shape[0], vector.shape[
    #     1]))

    # ==========================================================================
    # KMeansClusters
    # ==========================================================================
    # docs = list(corpus.title_tagged(fileids=subset_fileids))
    # observations = list(corpus.title_tagged(fileids=subset_fileids))
    #
    # # Text2FrequencyVector
    # prepare_data = Pipeline(
    #     [('normalize', TextNormalizer()),
    #      ('vectorize',
    #       Text2FrequencyVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # model = KMeansClusters(k=10)
    # labels = model.fit_transform(X)
    #
    # Cluster_Plotting.plot_clusters_2d(X, labels)

    # --------------------------------------------------------------------------
    # Text2OneHotVector

    # docs = list(corpus.title_tagged(fileids=subset_fileids))
    # observations = list(corpus.title_tagged(fileids=subset_fileids))
    # prepare_data = Pipeline([('normalize', TextNormalizer()),
    #                          ('vectorize',
    #                           Text2OneHotVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # model = KMeansClusters(k=10)
    # labels = model.fit_transform(X)
    #
    # Cluster_Plotting.plot_clusters_2d(X, labels)

    # --------------------------------------------------------------------------
    # Text2Doc2VecVector

    # docs = list(corpus.title_tagged(fileids=subset_fileids))
    # observations = list(corpus.title_tagged(fileids=subset_fileids))
    # prepare_data = Pipeline([('normalize', TextNormalizer()),
    #                          ('vectorize',
    #                           Text2Doc2VecVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # model = KMeansClusters(k=10)
    # labels = model.fit_transform(X)
    #
    # Cluster_Plotting.plot_clusters_2d(X, labels)

    # --------------------------------------------------------------------------
    # Text2TFIDVector

    # docs = list(corpus.title_tagged(fileids=subset_fileids))
    # observations = list(corpus.title_tagged(fileids=subset_fileids))
    # prepare_data = Pipeline([('normalize', TextNormalizer()),
    #                          ('vectorize',
    #                           Text2TFIDVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # model = KMeansClusters(k=10)
    # labels = model.fit_transform(X)
    #
    # Cluster_Plotting.plot_clusters_2d(X, labels)

    # ==========================================================================
    # HierarchicalClustering
    # ==========================================================================
    # Text2FrequencyVector

    # observations = list(corpus.title_tagged(fileids=subset_fileids))
    # prepare_data = Pipeline([('normalize', TextNormalizer()),
    #                          ('vectorize',
    #                           Text2FrequencyVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # for linkage in ('ward', 'average', 'complete', 'single'):
    #     clustering = HierarchicalClustering(linkage=linkage, n_clusters=10)
    #     t0 = time()
    #     labels = clustering.fit_transform(X)
    #     print("%s :\t%.2fs" % (linkage, time() - t0))
    #
    #     Cluster_Plotting.plot_clusters_2d(X, labels)

    # --------------------------------------------------------------------------
    # Text2OneHotVector
    #
    # observations = list(corpus.title_tagged(fileids=subset_fileids))
    # prepare_data = Pipeline(
    #     [('normalize', TextNormalizer()),
    #      ('vectorize',
    #       Text2OneHotVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # for linkage in ('ward', 'average', 'complete', 'single'):
    #     clustering = HierarchicalClustering(linkage=linkage, n_clusters=10)
    #     t0 = time()
    #     labels = clustering.fit_transform(X)
    #     print("%s :\t%.2fs" % (linkage, time() - t0))
    #
    #     Cluster_Plotting.plot_clusters_2d(X, labels)

    # --------------------------------------------------------------------------
    # Text2TFIDVector

    # observations = list(corpus.title_tagged(fileids=subset_fileids))
    #
    # prepare_data = Pipeline(
    #     [('normalize', TextNormalizer()),
    #      ('vectorize',
    #       Text2TFIDVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # for linkage in ('ward', 'average', 'complete', 'single'):
    #     clustering = HierarchicalClustering(linkage=linkage, n_clusters=10)
    #     t0 = time()
    #     labels = clustering.fit_transform(X)
    #     print("%s :\t%.2fs" % (linkage, time() - t0))
    #
    #     Cluster_Plotting.plot_clusters_2d(X, labels)

    # --------------------------------------------------------------------------
    # Text2Doc2VecVector

    # observations = list(corpus.title_tagged(fileids=subset_fileids))
    # prepare_data = Pipeline(
    #     [('normalize', TextNormalizer()),
    #      ('vectorize', Text2Doc2VecVector())])
    # X = prepare_data.fit_transform(observations)
    #
    # for linkage in ('ward', 'average', 'complete', 'single'):
    #     clustering = HierarchicalClustering(linkage=linkage, n_clusters=10)
    #     t0 = time()
    #     labels = clustering.fit_transform(X)
    #     print("%s :\t%.2fs" % (linkage, time() - t0))
    #
    #     Cluster_Plotting.plot_clusters_2d(X, labels)

    # ==========================================================================
    # SklearnTopicModels
    # ==========================================================================
    # """
    # options
    # LDA , LSA,  NMF
    # 'freqvec','tfidvec','onehotvec'
    # """
    # observations = list(corpus.title_tagged(fileids=subset_fileids))
    # model = SklearnTopicModels(n_topics=5, estimator='NMF',
    #                            vectorizor="tfidvec")
    #
    # model.fit_transform(observations)
    # topics = model.get_topics(n_term=10)
    # for topic, terms in topics.items():
    #     print("Topic #{} \t:".format(topic))
    #     print(terms)
