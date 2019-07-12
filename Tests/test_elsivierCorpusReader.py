from unittest import TestCase

from datetime import datetime

from CorpusReaders import Elsevier_Corpus_Reader


class TestRawCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corp = Elsevier_Corpus_Reader.RawCorpusReader(
            "Test_Corpus/Raw_corpus/")

    def test_docs(self):
        target = {'@_fa': 'true', 'link': [{'@_fa': 'true', '@ref': 'self', '@href': 'https://api.elsevier.com/content/abstract/scopus_id/84963627673'}, {'@_fa': 'true', '@ref': 'author-affiliation', '@href': 'https://api.elsevier.com/content/abstract/scopus_id/84963627673?field=author,affiliation'}, {'@_fa': 'true', '@ref': 'scopus', '@href': 'https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp=84963627673&origin=inward'}, {'@_fa': 'true', '@ref': 'scopus-citedby', '@href': 'https://www.scopus.com/inward/citedby.uri?partnerID=HzOxMe3b&scp=84963627673&origin=inward'}], 'prism:url': 'https://api.elsevier.com/content/abstract/scopus_id/84963627673', 'dc:identifier': 'SCOPUS_ID:84963627673', 'eid': '2-s2.0-84963627673', 'dc:title': 'Development of a hall-effect based skin sensor', 'dc:creator': 'Tomo T.', 'prism:publicationName': '2015 IEEE SENSORS - Proceedings', 'prism:isbn': [{'@_fa': 'true', '$': '9781479982028'}], 'prism:pageRange': None, 'prism:coverDate': '2015-12-31', 'prism:coverDisplayDate': '31 December 2015', 'prism:doi': '10.1109/ICSENS.2015.7370435', 'dc:description': "© 2015 IEEE. In this paper we introduce a prototype of a novel hall-effect based skin sensor for robotic applications. It uses a small sized chip that provides 3-axis digital output in a compact package. Our purpose was to evaluate the feasibility of measuring 3-axis force while maintain a soft exterior for safe interactions. Silicone was used to produce the soft skin layer with about 8 mm thickness. An MLX90393 chip was installed at the bottom of layer, with a small magnet approximately 5mm above it to measure 3-axial magnetic field data. To evaluate the sensor's performance, an experiment was conducted by measuring normal and shear force when applying total forces of 0.7-14N in the normal and tangential directions of the sensor. The test revealed that the sensor prototype was able to differentiate the components of the force vector, with limited crosstalk. A calibration was performed to convert the measurements of the magnetic field to force values.", 'citedby-count': '5', 'affiliation': [{'@_fa': 'true', 'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60023462', 'afid': '60023462', 'affilname': 'Waseda University', 'affiliation-city': 'Tokyo', 'affiliation-country': 'Japan'}, {'@_fa': 'true', 'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60004956', 'afid': '60004956', 'affilname': 'Instituto Superior Técnico', 'affiliation-city': 'Lisbon', 'affiliation-country': 'Portugal'}], 'prism:aggregationType': 'Conference Proceeding', 'subtype': 'cp', 'subtypeDescription': 'Conference Paper', 'author-count': {'@limit': '100', '$': '6'}, 'author': [{'@_fa': 'true', '@seq': '1', 'author-url': 'https://api.elsevier.com/content/author/author_id/57188759249', 'authid': '57188759249', 'authname': 'Tomo T.', 'surname': 'Tomo', 'given-name': 'Tito Pradhono', 'initials': 'T.P.', 'afid': [{'@_fa': 'true', '$': '60023462'}]}, {'@_fa': 'true', '@seq': '2', 'author-url': 'https://api.elsevier.com/content/author/author_id/56531989700', 'authid': '56531989700', 'authname': 'Somlor S.', 'surname': 'Somlor', 'given-name': 'Sophon', 'initials': 'S.', 'afid': [{'@_fa': 'true', '$': '60023462'}]}, {'@_fa': 'true', '@seq': '3', 'author-url': 'https://api.elsevier.com/content/author/author_id/36662716600', 'authid': '36662716600', 'authname': 'Schmitz A.', 'surname': 'Schmitz', 'given-name': 'Alexander', 'initials': 'A.', 'afid': [{'@_fa': 'true', '$': '60023462'}]}, {'@_fa': 'true', '@seq': '4', 'author-url': 'https://api.elsevier.com/content/author/author_id/35355828000', 'authid': '35355828000', 'authname': 'Hashimoto S.', 'surname': 'Hashimoto', 'given-name': 'Shuji', 'initials': 'S.', 'afid': [{'@_fa': 'true', '$': '60023462'}]}, {'@_fa': 'true', '@seq': '5', 'author-url': 'https://api.elsevier.com/content/author/author_id/7102227917', 'authid': '7102227917', 'authname': 'Sugano S.', 'surname': 'Sugano', 'given-name': 'Shigeki', 'initials': 'S.', 'afid': [{'@_fa': 'true', '$': '60023462'}]}, {'@_fa': 'true', '@seq': '6', 'author-url': 'https://api.elsevier.com/content/author/author_id/24479375600', 'authid': '24479375600', 'authname': 'Jamone L.', 'surname': 'Jamone', 'given-name': 'Lorenzo', 'initials': 'L.', 'afid': [{'@_fa': 'true', '$': '60004956'}]}], 'authkeywords': 'magnetic | sensor | skin | tactile', 'article-number': '7370435', 'source-id': '21100455562', 'fund-no': 'undefined', 'openaccess': '0', 'openaccessFlag': False}
        result = next(self.corp.docs())[0]
        self.assertEqual(result, target)


class TestScopusCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusCorpusReader(
            "Test_Corpus/Split_corpus/")

    def _helper_dict(self, gen_method, field):
        gen = self.corpus.docs()
        for result in gen_method:
            try:
                target = next(gen)[field]
            except KeyError:
                target = []
            self.assertEqual(target, result)
            self.assertEqual(list, type(result))

    def _helper_list(self, gen_method, field, attribute):
        gen = self.corpus.docs()
        for result in gen_method:
            try:
                _dict = next(gen)[field]
                target = [_item[attribute] for _item in
                          _dict]
            except KeyError:
                target = []
            self.assertEqual(target, result)
            self.assertEqual(len(target), len(result))
            self.assertEqual(list, type(result))

    def _helper_string(self, gen_method, field=None, attribute=None):
        targets = []
        for doc in self.corpus.docs():
            try:
                _dict = doc[field]
                if attribute is not None:
                    for _item in _dict:
                        if _item[attribute] is not None:
                            targets.append(_item[attribute])
                        else:
                            targets.append('unk')
                else:
                    if _dict is not None:
                        targets.append(_dict)
                    else:
                        targets.append('unk')
            except KeyError:
                targets.append('unk')

        results = list(gen_method)
        self.assertEqual(len(targets), len(results))
        for result, target in zip(results, targets):
            self.assertEqual(target, result)
            self.assertEqual(str, type(result))

    def test_affiliation_l(self):
        self._helper_dict(self.corpus.affiliation_l(),
                          'affiliation')

    def test_affiliation_city_l(self):
        self._helper_list(self.corpus.affiliation_city_l(),
                          'affiliation',
                          'affiliation-city')

    def test_affiliation_city_s(self):
        self._helper_string(self.corpus.affiliation_city_s(),
                            'affiliation',
                            'affiliation-city')

    def test_affiliation_country_l(self):
        self._helper_list(self.corpus.affiliation_country_l(),
                          'affiliation',
                          'affiliation-country')

    def test_affiliation_country_s(self):
        self._helper_string(self.corpus.affiliation_country_s(),
                            'affiliation',
                            'affiliation-country')

    def test_affiliation_url_l(self):
        self._helper_list(self.corpus.affiliation_url_l(),
                          'affiliation',
                          'affiliation-url')

    def test_affiliation_url_s(self):
        self._helper_string(self.corpus.affiliation_url_s(),
                            'affiliation',
                            'affiliation-url')

    def test_affiliation_name_l(self):
        self._helper_list(self.corpus.affiliation_name_l(),
                          'affiliation',
                          'affilname')

    def test_affiliation_name_s(self):
        self._helper_string(self.corpus.affiliation_name_s(),
                            'affiliation',
                            'affilname')

    def test_affiliation_id_l(self):
        self._helper_list(self.corpus.affiliation_id_l(),
                          'affiliation',
                          'afid')

    def test_affiliation_id_s(self):
        self._helper_string(self.corpus.affiliation_id_s(),
                            'affiliation',
                            'afid')

    def test_keywords_l(self):
        gen = self.corpus.docs()
        for result in self.corpus.keywords_l():
            try:
                target = [keyword.strip() for keyword in
                            next(gen)['authkeywords'].split("|")]
            except KeyError:
                target = []
            self.assertEqual(target, result)
            self.assertEqual(len(target), len(result))
            self.assertEqual(list, type(result))

    def test_keywords_string(self):
        gen = self.corpus.docs()
        for result in self.corpus.keywords_string():
            try:
                target = ' '.join([keyword.strip() for keyword in
                                   next(gen)[
                    'authkeywords'].split("|")])
            except KeyError:
                target = 'unk'
            self.assertEqual(target, result)
            self.assertEqual(len(target), len(result))
            self.assertEqual(str, type(result))

    def test_keywords_phrase(self):
        targets = []
        for doc in self.corpus.docs():
            try:
                target_list = [keyword.strip() for keyword in
                               doc['authkeywords'].split("|")]
            except KeyError:
                target_list = []

            if target_list:
                for phrase in target_list:
                    targets.append(phrase)
            else:
                targets.append('unk')

        results = list(self.corpus.keywords_phrase())
        self.assertEqual(len(targets), len(results))
        for result, target in zip(results, targets):
            self.assertEqual(target, result)
            self.assertEqual(str, type(result))

    def test_keywords_s(self):
        targets = []
        for doc in self.corpus.docs():
            try:
                target_list = [keyword.strip() for keyword in
                               doc['authkeywords'].split("|")]
            except KeyError:
                target_list = []

            if target_list:
                for phrase in target_list:
                    for word in phrase.split(" "):
                        targets.append(word.strip())
            else:
                targets.append('unk')

        results = list(self.corpus.keywords_s())
        self.assertEqual(len(targets), len(results))
        for result, target in zip(results, targets):
            self.assertEqual(target, result)
            self.assertEqual(str, type(result))

    def test_author_data_l(self):
        self._helper_dict(self.corpus.author_data_l(),
                          'author')

    def test_author_data_id_l(self):
        self._helper_list(self.corpus.author_data_id_l(),
                          'author',
                          'authid')

    def test_author_data_id_s(self):
        self._helper_string(self.corpus.author_data_id_s(),
                            'author',
                            'authid')

    def test_author_data_name_full_l(self):
        self._helper_list(self.corpus.author_data_name_full_l(),
                          'author',
                          'authname')

    def test_author_data_name_full_s(self):
        self._helper_string(self.corpus.author_data_name_full_s(),
                            'author',
                            'authname')

    def test_author_data_url_l(self):
        self._helper_list(self.corpus.author_data_url_l(),
                          'author',
                          'author-url')

    def test_author_data_url_s(self):
        self._helper_string(self.corpus.author_data_url_s(),
                            'author',
                            'author-url')

    def test_author_data_name_given_l(self):
        self._helper_list(self.corpus.author_data_name_given_l(),
                          'author',
                          'given-name')

    def test_author_data_name_given_s(self):
        self._helper_string(self.corpus.author_data_name_given_s(),
                            'author',
                            'given-name')

    def test_author_data_initial_l(self):
        self._helper_list(self.corpus.author_data_initial_l(),
                          'author',
                          'initials')

    def test_author_data_initial_s(self):
        self._helper_string(self.corpus.author_data_initial_s(),
                            'author',
                            'initials')

    def test_author_data_name_surname_l(self):
        self._helper_list(self.corpus.author_data_name_surname_l(),
                          'author',
                          'surname')

    def test_author_data_name_surname_s(self):
        self._helper_string(self.corpus.author_data_name_surname_s(),
                            'author',
                            'surname')

    def test_stat_num_authors(self):
        gen = self.corpus.docs()
        for result in self.corpus.stat_num_authors():
            try:
                target = int(next(gen)['author-count']["$"])
            except (KeyError, TypeError):
                target = 0
            self.assertEqual(target, result)
            self.assertEqual(int, type(result))

    def test_stat_num_citations(self):
        gen = self.corpus.docs()
        for result in self.corpus.stat_num_citations():
            try:
                target = int(next(gen)['citedby-count'])
            except (KeyError, TypeError):
                target = 0
            self.assertEqual(target, result)
            self.assertEqual(int, type(result))

    def test_identifier_scopus(self):
        self._helper_string(gen_method=self.corpus.identifier_scopus(),
                            field='dc:identifier', attribute=None)

    def test_identifier_electronic(self):
        self._helper_string(gen_method=self.corpus.identifier_electronic(),
                            field='eid', attribute=None)

    def test_identifier_doi(self):
        self._helper_string(gen_method=self.corpus.identifier_doi(),
                            field='prism:doi', attribute=None)

    def test_identifier_issn(self):
        self._helper_string(gen_method=self.corpus.identifier_issn(),
                            field='prism:issn', attribute=None)

    def test_identifier_pubmed(self):
        self._helper_string(gen_method=self.corpus.identifier_pubmed(),
                            field='pubmed-id', attribute=None)

    def test_identifier_source(self):
        self._helper_string(gen_method=self.corpus.identifier_source(),
                            field='source-id', attribute=None)

    def test_publication_type(self):
        self._helper_string(gen_method=self.corpus.publication_type(),
                            field='prism:aggregationType', attribute=None)

    def test_publication_name(self):
        self._helper_string(gen_method=self.corpus.publication_name(),
                            field='prism:publicationName', attribute=None)

    def test_publication_subtype(self):
        self._helper_string(gen_method=self.corpus.publication_subtype(),
                            field='subtypeDescription', attribute=None)

    def test_publication_volume(self):
        target = 77
        result = next(self.corpus.publication_volume())
        self.assertEqual(target, result)

    def test_publication_issue(self):
        target = 10
        result = next(self.corpus.publication_issue())
        self.assertEqual(target, result)

    def test_publication_pages(self):
        target = (1717, 1722)
        result = next(self.corpus.publication_pages())
        self.assertEqual(target, result)

    def test_publication_date(self):
        target = datetime(2006, 10, 1, 0, 0)
        result = next(self.corpus.publication_date())
        self.assertEqual(target, result)

    def test_publication_year(self):
        target = 2006
        result = next(self.corpus.publication_year())
        self.assertEqual(target, result)

    def test_document_title(self):
        self._helper_string(gen_method=self.corpus.document_title(),
                            field='dc:title', attribute=None)

    def test_document_description(self):
        self._helper_string(gen_method=self.corpus.document_description(),
                            field='dc:description', attribute=None)

# class TestScopusProcessedCorpusReader(TestCase):
#     def setUp(self) -> None:
#         self.corp = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
#             "Corpus/Processed_corpus/")
#
#     def test_title_sents(self):
#         target = ('Histologic', 'NNP')
#         result = next(self.corp.title_sents())[0]
#         self.assertEqual(result, target)
#
#     def test_title_tagged(self):
#         target = ('Histologic', 'NNP')
#         result = next(self.corp.title_tagged_word())
#         self.assertEqual(result, target)
#
#     def test_title_words(self):
#         target = 'Histologic'
#         result = next(self.corp.title_words())
#         self.assertEqual(result, target)
#
#     def test_describe(self):
#         target = {'files': 56350,
#                   'topics': 1,
#                   'titles': 56350,
#                   'words': 694742,
#                   'vocab': 33065,
#                   'lexdiv': 21.011401784364132,
#                   'tpdoc': 1.0,
#                   'wptit': 12.32905057675244}
#
#         result = self.corp.describe()
#         for metric in target:
#             self.assertEqual(result[metric], target[metric])
#
#     def test_ngrams(self):
#         gen = self.corp.ngrams(n=3)
#         result = next(gen)
#         target = ('<s>', '<s>', 'Histologic')
#         self.assertEqual(target, result)
#         gen = self.corp.ngrams(n=6)
#         result = next(gen)
#         target = ('<s>', '<s>', '<s>', '<s>', '<s>', 'Histologic')
#         self.assertEqual(target, result)
#
#
# class TestCorpuSubsetLoader(TestCase):
#     def setUp(self) -> None:
#         self.corp = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
#             "Corpus/Processed_corpus/")
#         self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corp,
#                                                               n_folds=12,
#                                                               shuffle=False)
#
#     def test_fileids(self):
#         target = 4696
#         result = len(next(self.loader.fileids(test=True)))
#         self.assertEqual(result, target)
#         target = 51654
#         result = len(next(self.loader.fileids(train=True)))
#         self.assertEqual(result, target)


