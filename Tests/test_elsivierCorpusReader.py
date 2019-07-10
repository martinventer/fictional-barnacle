from unittest import TestCase

from datetime import datetime

from CorpusReaders import Elsevier_Corpus_Reader


class TestRawCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corp = Elsevier_Corpus_Reader.RawCorpusReader(
            "Corpus/Raw_corpus/")

    def test_docs(self):
        target = [{'@_fa': 'true', 'link': [{'@_fa': 'true', '@ref': 'self', '@href': 'https://api.elsevier.com/content/abstract/scopus_id/85060282060'}, {'@_fa': 'true', '@ref': 'author-affiliation', '@href': 'https://api.elsevier.com/content/abstract/scopus_id/85060282060?field=author,affiliation'}, {'@_fa': 'true', '@ref': 'scopus', '@href': 'https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp=85060282060&origin=inward'}, {'@_fa': 'true', '@ref': 'scopus-citedby', '@href': 'https://www.scopus.com/inward/citedby.uri?partnerID=HzOxMe3b&scp=85060282060&origin=inward'}], 'prism:url': 'https://api.elsevier.com/content/abstract/scopus_id/85060282060', 'dc:identifier': 'SCOPUS_ID:85060282060', 'eid': '2-s2.0-85060282060', 'dc:title': 'Robots, productivity and quality', 'dc:creator': 'Rosen C.', 'prism:publicationName': 'Proceedings of the ACM Annual Conference, ACM 1972', 'prism:volume': '1', 'prism:pageRange': '47-57', 'prism:coverDate': '1972-08-01', 'prism:coverDisplayDate': '1 August 1972', 'dc:description': '© Association for Computing Machinery, Inc 1972. All rights reserved. There is a growing national need to increase the real productivity of our society, wherein "productivity" is redefined to include such major factors as the quality of life of workers and the quality of products, consistent with the desires and expectations of the general public. This paper proposed the development of automation technology designed to increase quality, in all its aspects, at an acceptable cost to society. The proposed program is divided into two phases. The first phase is designed to catalyze the potential resources of industrial concerns by developing two demonstrable systems that include generalpurpose programmed manipulation and automated inspection. The second phase, with longer term objectives, would aim at devising techniques to broaden the utilization of programmed manipulators and sensors, to provide supervisory control of these systems by human speech, and to develop a capability for automatic manipulation of two or more sensor-controlled "hands" working cooperatively.', 'citedby-count': '2', 'affiliation': [{'@_fa': 'true', 'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60000461', 'afid': '60000461', 'affilname': 'SRI International', 'affiliation-city': 'Menlo Park', 'affiliation-country': 'United States'}], 'prism:aggregationType': 'Conference Proceeding', 'subtype': 'cp', 'subtypeDescription': 'Conference Paper', 'author-count': {'@limit': '100', '$': '1'}, 'author': [{'@_fa': 'true', '@seq': '1', 'author-url': 'https://api.elsevier.com/content/author/author_id/7202592484', 'authid': '7202592484', 'authname': 'Rosen C.', 'surname': 'Rosen', 'given-name': 'Charles A.', 'initials': 'C.A.', 'afid': [{'@_fa': 'true', '$': '60000461'}]}], 'source-id': '21100903188', 'fund-no': 'undefined', 'openaccess': '0', 'openaccessFlag': False}, {'@_fa': 'true', 'link': [{'@_fa': 'true', '@ref': 'self', '@href': 'https://api.elsevier.com/content/abstract/scopus_id/0015346272'}, {'@_fa': 'true', '@ref': 'author-affiliation', '@href': 'https://api.elsevier.com/content/abstract/scopus_id/0015346272?field=author,affiliation'}, {'@_fa': 'true', '@ref': 'scopus', '@href': 'https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp=0015346272&origin=inward'}, {'@_fa': 'true', '@ref': 'scopus-citedby', '@href': 'https://www.scopus.com/inward/citedby.uri?partnerID=HzOxMe3b&scp=0015346272&origin=inward'}], 'prism:url': 'https://api.elsevier.com/content/abstract/scopus_id/0015346272', 'dc:identifier': 'SCOPUS_ID:0015346272', 'eid': '2-s2.0-0015346272', 'dc:title': 'Review of materials processing literature 1966-1968: 4—plastic working of metals', 'dc:creator': 'Boulger F.', 'prism:publicationName': 'Journal of Manufacturing Science and Engineering, Transactions of the ASME', 'prism:issn': '10871357', 'prism:eIssn': '15288935', 'prism:volume': '94', 'prism:issueIdentifier': '2', 'prism:pageRange': '721-725', 'prism:coverDate': '1972-01-01', 'prism:coverDisplayDate': 'May 1972', 'prism:doi': '10.1115/1.3428235', 'citedby-count': '0', 'affiliation': [{'@_fa': 'true', 'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60033479', 'afid': '60033479', 'affilname': 'Battelle Columbus Lab', 'affiliation-city': 'Columbus', 'affiliation-country': 'United States'}], 'prism:aggregationType': 'Journal', 'subtype': 're', 'subtypeDescription': 'Review', 'author-count': {'@limit': '100', '$': '1'}, 'author': [{'@_fa': 'true', '@seq': '1', 'author-url': 'https://api.elsevier.com/content/author/author_id/6602887172', 'authid': '6602887172', 'authname': 'Boulger F.', 'surname': 'Boulger', 'given-name': 'F. W.', 'initials': 'F.W.', 'afid': [{'@_fa': 'true', '$': '60033479'}]}], 'source-id': '20966', 'fund-no': 'undefined', 'openaccess': '0', 'openaccessFlag': False}]
        result = next(self.corp.docs())
        print(result)
        self.assertEqual(result, target)


class TestScopusCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corpus = Elsevier_Corpus_Reader.ScopusCorpusReader(
            "Corpus/Split_corpus/")

    def test_affiliation_l(self):
        target = [{'@_fa': 'true',
                   'affiliation-city': 'Augusta',
                   'affiliation-country': 'United States',
                   'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60024399',
                   'affilname': 'Medical College of Georgia',
                   'afid': '60024399'},
                   {'@_fa': 'true',
                  'affiliation-city': 'Los Angeles',
                    'affiliation-country': 'United States',
                    'affiliation-url':
                        'https://api.elsevier.com/content/affiliation/affiliation_id/60019009',
                    'affilname': 'University of Southern California, School of Dentistry',
                    'afid': '60019009'},
                    {'@_fa': 'true',
                    'affiliation-city': 'Gothenburg',
                    'affiliation-country': 'Sweden',
                    'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60016437',
                    'affilname': 'Göteborgs Universitet',
                    'afid': '60016437'},
                    {'@_fa': 'true',
                    'affiliation-city': 'Seattle',
                    'affiliation-country': 'United States',
                    'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60015481',
                    'affilname': 'University of Washington, Seattle',
                    'afid': '60015481'},
                    {'@_fa': 'true',
                    'affiliation-city': 'Jerusalem',
                    'affiliation-country': 'Israel',
                    'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60007903',
                    'affilname': 'Hebrew University of Jerusalem',
                    'afid': '60007903'}]
        result = next(self.corpus.affiliation_l())
        self.assertEqual(target, result)

    def test_affiliation_city_l(self):
        target = ['Augusta', 'Los Angeles', 'Gothenburg', 'Seattle', 'Jerusalem']
        result = next(self.corpus.affiliation_city_l())
        self.assertEqual(target, result)

    def test_affiliation_city_s(self):
        target = 'Augusta'
        result = next(self.corpus.affiliation_city_s())
        self.assertEqual(target, result)

    def test_affiliation_country_l(self):
        target = ['United States', 'United States', 'Sweden', 'United States', 'Israel']
        result = next(self.corpus.affiliation_country_l())
        self.assertEqual(target, result)

    def test_affiliation_country_s(self):
        target = 'United States'
        result = next(self.corpus.affiliation_country_s())
        self.assertEqual(target, result)

    def test_affiliation_url_l(self):
        target = ['https://api.elsevier.com/content/affiliation/affiliation_id/60024399',
                     'https://api.elsevier.com/content/affiliation/affiliation_id/60019009',
                     'https://api.elsevier.com/content/affiliation/affiliation_id/60016437',
                     'https://api.elsevier.com/content/affiliation/affiliation_id/60015481',
                     'https://api.elsevier.com/content/affiliation/affiliation_id/60007903']
        result = next(self.corpus.affiliation_url_l())
        self.assertEqual(target, result)

    def test_affiliation_url_s(self):
        target = 'https://api.elsevier.com/content/affiliation/affiliation_id/60024399'
        result = next(self.corpus.affiliation_url_s())
        self.assertEqual(target, result)

    def test_affiliation_name_l(self):
        target = ['Medical College of Georgia',
                     'University of Southern California, School of Dentistry',
                     'Göteborgs Universitet',
                     'University of Washington, Seattle',
                     'Hebrew University of Jerusalem']
        result = next(self.corpus.affiliation_name_l())
        self.assertEqual(target, result)

    def test_affiliation_name_s(self):
        target = 'Medical College of Georgia'
        result = next(self.corpus.affiliation_name_s())
        self.assertEqual(target, result)

    def test_affiliation_id_l(self):
        target = ['60024399', '60019009', '60016437', '60015481', '60007903']
        result = next(self.corpus.affiliation_id_l())
        self.assertEqual(target, result)

    def test_affiliation_id_s(self):
        target = '60024399'
        result = next(self.corpus.affiliation_id_s())
        self.assertEqual(target, result)

    def test_keywords_l(self):
        target = ['Dental implants', 'Histology', 'Osseointegration', 'Surgery']
        result = next(self.corpus.keywords_l())
        self.assertEqual(target, result)

    def test_keywords_string(self):
        target = 'Dental implants Histology Osseointegration Surgery'
        result = next(self.corpus.keywords_string())
        self.assertEqual(target, result)

    def test_keywords_phrase(self):
        target = 'Dental implants'
        result = next(self.corpus.keywords_phrase())
        self.assertEqual(target, result)

    def test_keywords_s(self):
        target = 'Dental'
        result = next(self.corpus.keywords_s())
        self.assertEqual(target, result)

    def test_author_data_l(self):
        target = [{'@_fa': 'true',
                      '@seq': '1',
                      'afid': [{'$': '60019009', '@_fa': 'true'},
                               {'$': '60015481', '@_fa': 'true'}],
                      'authid': '55547124100',
                      'authname': 'Becker W.',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/55547124100',
                      'given-name': 'William',
                      'initials': 'W.',
                      'surname': 'Becker'},
                     {'@_fa': 'true',
                      '@seq': '2',
                      'afid': [{'$': '60024399', '@_fa': 'true'}],
                      'authid': '7007040311',
                      'authname': 'Wikesjö U.',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/7007040311',
                      'given-name': 'Ulf M.E.',
                      'initials': 'U.M.E.',
                      'surname': 'Wikesjö'},
                     {'@_fa': 'true',
                      '@seq': '3',
                      'afid': [{'$': '60016437', '@_fa': 'true'}],
                      'authid': '7005071364',
                      'authname': 'Sennerby L.',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/7005071364',
                      'given-name': 'Lars',
                      'initials': 'L.',
                      'surname': 'Sennerby'},
                     {'@_fa': 'true',
                      '@seq': '4',
                      'afid': [{'$': '60024399', '@_fa': 'true'}],
                      'authid': '6507418137',
                      'authname': 'Qahash M.',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/6507418137',
                      'given-name': 'Mohammed',
                      'initials': 'M.',
                      'surname': 'Qahash'},
                     {'@_fa': 'true',
                      '@seq': '5',
                      'afid': [{'$': '60007903', '@_fa': 'true'}],
                      'authid': '7006267759',
                      'authname': 'Hujoel P.',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/7006267759',
                      'given-name': 'Philippe',
                      'initials': 'P.',
                      'surname': 'Hujoel'},
                     {'@_fa': 'true',
                      '@seq': '6',
                      'authid': '7403014169',
                      'authname': 'Goldstein M.',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/7403014169',
                      'given-name': 'Moshe',
                      'initials': 'M.',
                      'surname': 'Goldstein'},
                     {'@_fa': 'true',
                      '@seq': '7',
                      'afid': [{'$': '60016437', '@_fa': 'true'}],
                      'authid': '15052667100',
                      'authname': 'Turkyilmaz I.',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/15052667100',
                      'given-name': 'Ilser',
                      'initials': 'I.',
                      'surname': 'Turkyilmaz'}]
        result = next(self.corpus.author_data_l())
        self.assertEqual(target, result)

    def test_author_data_id_l(self):
        target = ['55547124100',
                     '7007040311',
                     '7005071364',
                     '6507418137',
                     '7006267759',
                     '7403014169',
                     '15052667100']
        result = next(self.corpus.author_data_id_l())
        self.assertEqual(target, result)

    def test_author_data_id_s(self):
        target = '55547124100'
        result = next(self.corpus.author_data_id_s())
        self.assertEqual(target, result)

    def test_author_data_name_full_l(self):
        target = ['Becker W.',
                     'Wikesjö U.',
                     'Sennerby L.',
                     'Qahash M.',
                     'Hujoel P.',
                     'Goldstein M.',
                     'Turkyilmaz I.']
        result = next(self.corpus.author_data_name_full_l())
        self.assertEqual(target, result)

    def test_author_data_name_full_s(self):
        target = 'Becker W.'
        result = next(self.corpus.author_data_name_full_s())
        self.assertEqual(target, result)

    def test_author_data_url_l(self):
        target = ['https://api.elsevier.com/content/author/author_id/55547124100',
                     'https://api.elsevier.com/content/author/author_id/7007040311',
                     'https://api.elsevier.com/content/author/author_id/7005071364',
                     'https://api.elsevier.com/content/author/author_id/6507418137',
                     'https://api.elsevier.com/content/author/author_id/7006267759',
                     'https://api.elsevier.com/content/author/author_id/7403014169',
                     'https://api.elsevier.com/content/author/author_id/15052667100']
        result = next(self.corpus.author_data_url_l())
        self.assertEqual(target, result)

    def test_author_data_url_s(self):
        target = 'https://api.elsevier.com/content/author/author_id/55547124100'
        result = next(self.corpus.author_data_url_s())
        self.assertEqual(target, result)

    def test_author_data_name_given_l(self):
        target = ['William', 'Ulf M.E.', 'Lars', 'Mohammed', 'Philippe', 'Moshe', 'Ilser']
        result = next(self.corpus.author_data_name_given_l())
        self.assertEqual(target, result)

    def test_author_data_name_given_s(self):
        target = 'William'
        result = next(self.corpus.author_data_name_given_s())
        self.assertEqual(target, result)

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


