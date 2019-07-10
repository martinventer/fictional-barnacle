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
        print(result)
        self.assertEqual(result, target)


class TestScopusCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusCorpusReader(
            "Test_Corpus/Split_corpus/")
        # self.gen = self.corpus.docs()
        # self.trial_doc = next(self.gen)
        # self.trial_doc_a = next(self.gen)
        # self.trial_doc_b = next(self.gen)

    def test_affiliation_l(self):
        gen = self.corpus.docs()
        for result in self.corpus.affiliation_l():
            try:
                target = next(gen)['affiliation']
            except KeyError:
                target = []
            self.assertEqual(target, result)
            self.assertEqual(list, type(result))

    def test_affiliation_city_l(self):
        gen = self.corpus.docs()
        for result in self.corpus.affiliation_city_l():
            try:
                target = next(gen)['affiliation']
                target2 = target[0]['affiliation-city']
            except KeyError:
                target = []

            self.assertEqual(len(target), len(result))
            self.assertEqual(target2, result[0])
            self.assertEqual(list, type(result))

    def test_affiliation_city_s(self):
        target = self.trial_doc['affiliation'][0]['affiliation-city']
        result = next(self.corpus.affiliation_city_s())
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_affiliation_country_l(self):
        target = self.trial_doc['affiliation'][0]['affiliation-country']
        result = next(self.corpus.affiliation_country_l())
        self.assertEqual(target, result[0])
        target2 = len(self.trial_doc['affiliation'])
        self.assertEqual(target2, len(result))
        self.assertEqual(list, type(result))

    def test_affiliation_country_s(self):
        target = self.trial_doc['affiliation'][0]['affiliation-country']
        result = next(self.corpus.affiliation_country_s())
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_affiliation_url_l(self):
        target = self.trial_doc['affiliation'][0]['affiliation-url']
        result = next(self.corpus.affiliation_url_l())
        self.assertEqual(target, result[0])
        target2 = len(self.trial_doc['affiliation'])
        self.assertEqual(target2, len(result))
        self.assertEqual(list, type(result))

    def test_affiliation_url_s(self):
        target = self.trial_doc['affiliation'][0]['affiliation-url']
        result = next(self.corpus.affiliation_url_s())
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_affiliation_name_l(self):
        target = self.trial_doc['affiliation'][0]['affilname']
        result = next(self.corpus.affiliation_name_l())
        self.assertEqual(target, result[0])
        target2 = len(self.trial_doc['affiliation'])
        self.assertEqual(target2, len(result))
        self.assertEqual(list, type(result))

    def test_affiliation_name_s(self):
        target = self.trial_doc['affiliation'][0]['affilname']
        result = next(self.corpus.affiliation_name_s())
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_affiliation_id_l(self):
        target = self.trial_doc['affiliation'][0]['afid']
        result = next(self.corpus.affiliation_id_l())
        self.assertEqual(target, result[0])
        target2 = len(self.trial_doc['affiliation'])
        self.assertEqual(target2, len(result))
        self.assertEqual(list, type(result))

    def test_affiliation_id_s(self):
        target = self.trial_doc['affiliation'][0]['afid']
        result = next(self.corpus.affiliation_id_s())
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_keywords_l(self):
        target = [keyword.strip() for keyword in self.trial_doc_a[
            'authkeywords'].split("|")]
        gen = self.corpus.keywords_l()
        _ = next(gen)
        result = next(gen)
        self.assertEqual(target, result)
        self.assertEqual(list, type(result))

    def test_keywords_string(self):
        target = ' '.join([keyword.strip() for keyword in self.trial_doc_a[
            'authkeywords'].split("|")])
        gen = self.corpus.keywords_string()
        _ = next(gen)
        result = next(gen)
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_keywords_phrase(self):
        target = [keyword.strip() for keyword in self.trial_doc_a[
            'authkeywords'].split("|")][0]
        gen = self.corpus.keywords_phrase()
        _ = next(gen)
        result = next(gen)
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_keywords_s(self):
        target = [keyword.strip() for keyword in self.trial_doc_a[
            'authkeywords'].split("|")][0].split(' ')[0]
        gen = self.corpus.keywords_s()
        _ = next(gen)
        result = next(gen)
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_author_data_l(self):
        target = self.trial_doc['author']
        result = next(self.corpus.author_data_l())
        self.assertEqual(target, result)
        target2 = len(self.trial_doc['author'])
        self.assertEqual(target2, len(result))
        self.assertEqual(list, type(result))

    def test_author_data_id_l(self):
        target = [auth['authid'] for auth in self.trial_doc['author']]
        result = next(self.corpus.author_data_id_l())
        self.assertEqual(target, result)
        self.assertEqual(list, type(result))

    def test_author_data_id_s(self):
        target = self.trial_doc['author'][0]['authid']
        result = next(self.corpus.author_data_id_s())
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_author_data_name_full_l(self):
        target = [auth['authname'] for auth in self.trial_doc['author']]
        result = next(self.corpus.author_data_name_full_l())
        self.assertEqual(target, result)
        self.assertEqual(list, type(result))

    def test_author_data_name_full_s(self):
        target = self.trial_doc['author'][0]['authname']
        result = next(self.corpus.author_data_name_full_s())
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_author_data_url_l(self):
        target = [auth['author-url'] for auth in self.trial_doc['author']]
        result = next(self.corpus.author_data_url_l())
        self.assertEqual(target, result)
        self.assertEqual(list, type(result))

    def test_author_data_url_s(self):
        target = self.trial_doc['author'][0]['author-url']
        result = next(self.corpus.author_data_url_s())
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_author_data_name_given_l(self):
        target = [auth['given-name'] for auth in self.trial_doc['author']]
        result = next(self.corpus.author_data_name_given_l())
        self.assertEqual(target, result)
        self.assertEqual(list, type(result))

    def test_author_data_name_given_s(self):
        target = self.trial_doc['author'][0]['given-name']
        result = next(self.corpus.author_data_name_given_s())
        self.assertEqual(target, result)
        self.assertEqual(str, type(result))

    def test_author_data_initial_l(self):
        target = ['W.', 'U.M.E.', 'L.', 'M.', 'P.', 'M.', 'I.']
        result = next(self.corpus.author_data_initial_l())
        self.assertEqual(target, result)

    def test_author_data_initial_s(self):
        target = 'W.'
        result = next(self.corpus.author_data_initial_s())
        self.assertEqual(target, result)

    def test_author_data_name_surname_l(self):
        target = ['Becker', 'Wikesjö', 'Sennerby', 'Qahash', 'Hujoel', 'Goldstein', 'Turkyilmaz']
        result = next(self.corpus.author_data_name_surname_l())
        self.assertEqual(target, result)

    def test_author_data_name_surname_s(self):
        target = 'Becker'
        result = next(self.corpus.author_data_name_surname_s())
        self.assertEqual(target, result)

    def test_stat_num_authors(self):
        target = 7
        result = next(self.corpus.stat_num_authors())
        self.assertEqual(target, result)

    def test_stat_num_citations(self):
        target = 56
        result = next(self.corpus.stat_num_citations())
        self.assertEqual(target, result)

    def test_identifier_scopus(self):
        target = 'SCOPUS_ID:33750589187'
        result = next(self.corpus.identifier_scopus())
        self.assertEqual(target, result)

    def test_identifier_electronic(self):
        target = '2-s2.0-33750589187'
        result = next(self.corpus.identifier_electronic())
        self.assertEqual(target, result)

    def test_identifier_link(self):
        target = 'https://api.elsevier.com/content/abstract/scopus_id/33750589187'
        result = next(self.corpus.identifier_link())
        self.assertEqual(target, result)

    def test_identifier_doi(self):
        target = '10.1902/jop.2006.060090'
        result = next(self.corpus.identifier_doi())
        self.assertEqual(target, result)

    def test_identifier_issn(self):
        target = '00223492'
        result = next(self.corpus.identifier_issn())
        self.assertEqual(target, result)

    def test_identifier_pubmed(self):
        target = '17032115'
        result = next(self.corpus.identifier_pubmed())
        self.assertEqual(target, result)

    def test_identifier_source(self):
        target = '26173'
        result = next(self.corpus.identifier_source())
        self.assertEqual(target, result)

    def test_publication_type(self):
        target = 'Journal'
        result = next(self.corpus.publication_type())
        self.assertEqual(target, result)

    def test_publication_subtype(self):
        target = 'Article'
        result = next(self.corpus.publication_subtype())
        self.assertEqual(target, result)

    def test_publication_name(self):
        target = 'Journal of Periodontology'
        result = next(self.corpus.publication_name())
        self.assertEqual(target, result)

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
        target = 'Histologic evaluation of implants following flapless and flapped surgery: A study in canines'
        result = next(self.corpus.document_title())
        self.assertEqual(target, result)

    def test_document_description(self):
        target = 'Background: Flapless surgery requires penetration of the alveolar mucosa and bone without reflection of mucoperiosteal flaps. Do these techniques force gingival tissue or foreign materials into osteotomies? If so, do such tissues or materials interfere with osseointegration? A proof-of-principle study using a canine model attempted to answer these questions. Methods: Five young adult Hound Labrador mongrel dogs received implants with a moderately roughened surface by anodic oxidation using flapless or conventional one-stage (control) surgery in contralateral jaw quadrants. The implants were placed into the osteotomies, and the international stability quotient (ISQ) was recorded using resonance frequency analysis. These measurements were repeated following a 3-month healing interval when the animals were euthanized, and implants and surrounding tissues were retrieved and processed for histologic analysis. Results: The implants were stable upon insertion and demonstrated increased stability at 3 months without significant differences between surgical protocols. The histologic evaluation showed high bone-implant contact (flapless surgery: 54.7% ± 8.4%; control: 52.2% ± 13.0%; P >0.05) without evidence of gingival tissue or foreign body inclusions. There were no significant differences in marginal bone levels between the surgical protocols. Post-insertion and at 3 months, ISQ values depended on the amount of torque delivered. Immediately post-insertion, for every 1-unit increase in torque value, the ISQ increased by 0.3 (95% confidence interval: 0.1 to 0.4; P = 0.0043). Three months postoperatively, for every one-unit increase in torque the ISQ value decreased 0.2 (95% confidence interval: -0.4 to -0.1; P = 0.00 12). The effect of torque on ISQ values was independent of treatment effects and remained significant after adjustment for treatment. Conclusions: The results suggest that implants placed without flap reflection remain stable and exhibit clinically relevant osseointegration similar to when implants are placed with flapped procedures. Greater torque at implant placement resulted in less implant stability at 3 months.'
        result = next(self.corpus.document_description())
        self.assertEqual(target, result)

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


