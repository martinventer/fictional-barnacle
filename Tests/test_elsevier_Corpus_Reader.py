from unittest import TestCase

from CorpusReader import Elsevier_Corpus_Reader
from datetime import datetime


class TestScopusRawCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corp = Elsevier_Corpus_Reader.ScopusRawCorpusReader(
            "Tests/Corpus/Processed_corpus/")

    def test_docs(self):
        target = {'@_fa': 'true',
                  'link': [{'@_fa': 'true',
                               '@ref': 'self',
                               '@href': 'https://api.elsevier.com/content/abstract/scopus_id/85060282060'},
                              {'@_fa': 'true',
                               '@ref': 'author-affiliation',
                               '@href': 'https://api.elsevier.com/content/abstract/scopus_id/85060282060?field=author,affiliation'},
                              {'@_fa': 'true',
                               '@ref': 'scopus',
                               '@href': 'https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp=85060282060&origin=inward'},
                              {'@_fa': 'true',
                               '@ref': 'scopus-citedby',
                               '@href': 'https://www.scopus.com/inward/citedby.uri?partnerID=HzOxMe3b&scp=85060282060&origin=inward'}],
                     'prism:url': 'https://api.elsevier.com/content/abstract/scopus_id/85060282060',
                     'dc:identifier': 'SCOPUS_ID:85060282060',
                     'eid': '2-s2.0-85060282060',
                     'dc:title': 'Robots, productivity and quality',
                     'dc:creator': 'Rosen C.',
                     'prism:publicationName': 'Proceedings of the ACM Annual Conference, ACM 1972',
                     'prism:volume': '1',
                     'prism:pageRange': '47-57',
                     'prism:coverDate': '1972-08-01',
                     'prism:coverDisplayDate': '1 August 1972',
                     'dc:description': '© Association for Computing Machinery, Inc 1972. All rights reserved. There is a growing national need to increase the real productivity of our society, wherein "productivity" is redefined to include such major factors as the quality of life of workers and the quality of products, consistent with the desires and expectations of the general public. This paper proposed the development of automation technology designed to increase quality, in all its aspects, at an acceptable cost to society. The proposed program is divided into two phases. The first phase is designed to catalyze the potential resources of industrial concerns by developing two demonstrable systems that include generalpurpose programmed manipulation and automated inspection. The second phase, with longer term objectives, would aim at devising techniques to broaden the utilization of programmed manipulators and sensors, to provide supervisory control of these systems by human speech, and to develop a capability for automatic manipulation of two or more sensor-controlled "hands" working cooperatively.',
                     'citedby-count': '2',
                     'affiliation': [{'@_fa': 'true',
                                      'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60000461',
                                      'afid': '60000461',
                                      'affilname': 'SRI International',
                                      'affiliation-city': 'Menlo Park',
                                      'affiliation-country': 'United States'}],
                     'prism:aggregationType': 'Conference Proceeding',
                     'subtype': 'cp',
                     'subtypeDescription': 'Conference Paper',
                     'author-count': {'@limit': '100', '$': '1'},
                     'author': [{'@_fa': 'true',
                                 '@seq': '1',
                                 'author-url': 'https://api.elsevier.com/content/author/author_id/7202592484',
                                 'authid': '7202592484',
                                 'authname': 'Rosen C.',
                                 'surname': 'Rosen',
                                 'given-name': 'Charles A.',
                                 'initials': 'C.A.',
                                 'afid': [{'@_fa': 'true', '$': '60000461'}]}],
                     'source-id': '21100903188',
                     'fund-no': 'undefined',
                     'openaccess': '0',
                     'openaccessFlag': False,
                     'struct:title': [[('Robots', 'NNS'),
                                       (',', ','),
                                       ('productivity', 'NN'),
                                       ('and', 'CC'),
                                       ('quality', 'NN')]]}
        self.assertEqual(next(self.corp.docs()), target)

    def test_title_raw(self):
        target = 'Robots, productivity and quality'
        self.assertEqual(next(self.corp.title_raw()), target)

    def test_abstracts(self):
        target = '© Association for Computing Machinery, Inc 1972. All rights reserved. There is a growing national need to increase the real productivity of our society, wherein "productivity" is redefined to include such major factors as the quality of life of workers and the quality of products, consistent with the desires and expectations of the general public. This paper proposed the development of automation technology designed to increase quality, in all its aspects, at an acceptable cost to society. The proposed program is divided into two phases. The first phase is designed to catalyze the potential resources of industrial concerns by developing two demonstrable systems that include generalpurpose programmed manipulation and automated inspection. The second phase, with longer term objectives, would aim at devising techniques to broaden the utilization of programmed manipulators and sensors, to provide supervisory control of these systems by human speech, and to develop a capability for automatic manipulation of two or more sensor-controlled "hands" working cooperatively.'
        self.assertEqual(next(self.corp.abstracts()), target)

    def test_doc_ids(self):
        target = 'https://api.elsevier.com/content/abstract/scopus_id/85060282060'
        self.assertEqual(next(self.corp.doc_ids()), target)
        target = 'SCOPUS_ID:85060282060'
        self.assertEqual(next(self.corp.doc_ids(form='dc:identifier')), target)
        target = 'SCOPUS_ID:85060282060'
        self.assertEqual(next(self.corp.doc_ids(form='dc:identifier')), target)
        target = '2-s2.0-85060282060'
        self.assertEqual(next(self.corp.doc_ids(form='eid')), target)
        target = ''
        self.assertEqual(next(self.corp.doc_ids(form='prism:isbn')), target)
        target = ''
        self.assertEqual(next(self.corp.doc_ids(form='prism:doi')), target)
        target = ''
        self.assertEqual(next(self.corp.doc_ids(form='article-number')), target)

    def test_publication(self):
        target = 'Proceedings of the ACM Annual Conference, ACM 1972'
        self.assertEqual(next(self.corp.publication()), target)
        target = 'Conference Paper'
        self.assertEqual(next(self.corp.publication(form='subtypeDescription')),
                         target)
        target = 'Conference Proceeding'
        self.assertEqual(next(
            self.corp.publication(form='prism:aggregationType')), target)
        target = 'cp'
        self.assertEqual(next(self.corp.publication(form='subtype')), target)

    def test_pub_date(self):
        target = datetime(1972, 8, 1, 0, 0)
        self.assertEqual(next(self.corp.pub_date()), target)
        target = 1972
        self.assertEqual(next(self.corp.pub_date(form='year')), target)

    def test_author_data(self):
        target = [{'@_fa': 'true',
                      '@seq': '1',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/7202592484',
                      'authid': '7202592484',
                      'authname': 'Rosen C.',
                      'surname': 'Rosen',
                      'given-name': 'Charles A.',
                      'initials': 'C.A.',
                      'afid': [{'@_fa': 'true', '$': '60000461'}]}]
        self.assertEqual(next(self.corp.author_data()), target)

    def test_author_count(self):
        target = 1
        self.assertEqual(next(self.corp.author_count()), target)

    def test_author_name(self):
        target = 'Rosen C.'
        self.assertEqual(next(self.corp.author_name()), target)
        target = 'https://api.elsevier.com/content/author/author_id/7202592484'
        self.assertEqual(next(self.corp.author_name(form='author-url')),
                         target)
        target = '7202592484'
        self.assertEqual(next(self.corp.author_name(form='authid')), target)
        target = 'Rosen'
        self.assertEqual(next(self.corp.author_name(form='surname')), target)
        target = 'Charles A.'
        self.assertEqual(next(self.corp.author_name(form='given-name')),
                         target)
        target = 'C.A.'
        self.assertEqual(next(self.corp.author_name(form='initials')), target)

    def test_author_list(self):
        target = ['Rosen C.']
        self.assertEqual(next(self.corp.author_list()), target)
        target = ['https://api.elsevier.com/content/author/author_id/7202592484']
        self.assertEqual(next(self.corp.author_list(form='author-url')),
                         target)
        target = ['7202592484']
        self.assertEqual(next(self.corp.author_list(form='authid')), target)
        target = ['Rosen']
        self.assertEqual(next(self.corp.author_list(form='surname')), target)
        target = ['Charles A.']
        self.assertEqual(next(self.corp.author_list(form='given-name')),
                         target)
        target = ['C.A.']
        self.assertEqual(next(self.corp.author_list(form='initials')), target)

    def test_author_keyword_list(self):
        target = []
        self.assertEqual(next(self.corp.author_keyword_list()), target)

    def test_author_keyword(self):
        target = 'assembly'
        self.assertEqual(next(self.corp.author_keyword()), target)

    def test_doc_volume(self):
        target = '1'
        self.assertEqual(next(self.corp.doc_volume()), target)

    def test_doc_page_range(self):
        target = (47, 57)
        self.assertEqual(next(self.corp.doc_page_range()), target)

    def test_doc_citation_number(self):
        target = 2
        self.assertEqual(next(self.corp.doc_citation_number()), target)

    def test_affiliation_list(self):
        target = [{'@_fa': 'true',
                      'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60000461',
                      'afid': '60000461',
                      'affilname': 'SRI International',
                      'affiliation-city': 'Menlo Park',
                      'affiliation-country': 'United States'}]
        self.assertEqual(next(self.corp.affiliation_list()), target)


class TestScopusProcessedCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corp = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Tests/Corpus/Processed_corpus/")

    def test_title_sents(self):
        self.assertEqual(next(self.corp.title_sents())[0],
                         ('Robots', 'NNS'))

    def test_title_tagged(self):
        self.assertEqual(next(self.corp.title_tagged()),
                         ('Robots', 'NNS'))

    def test_title_words(self):
        self.assertEqual(next(self.corp.title_words()),
                         'Robots')

