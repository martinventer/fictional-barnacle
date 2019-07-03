from unittest import TestCase

from CorpusReader import Elsevier_Corpus_Reader
from datetime import datetime


class TestScopusRawCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corp = Elsevier_Corpus_Reader.ScopusRawCorpusReader(
            "Corpus/Processed_corpus/")

    def test_docs(self):
        target = {'@_fa': 'true',
                     'link': [{'@_fa': 'true',
                       '@ref': 'self',
                       '@href': 'https://api.elsevier.com/content/abstract/scopus_id/33750589187'},
                      {'@_fa': 'true',
                       '@ref': 'author-affiliation',
                       '@href': 'https://api.elsevier.com/content/abstract/scopus_id/33750589187?field=author,affiliation'},
                      {'@_fa': 'true',
                       '@ref': 'scopus',
                       '@href': 'https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp=33750589187&origin=inward'},
                      {'@_fa': 'true',
                       '@ref': 'scopus-citedby',
                       '@href': 'https://www.scopus.com/inward/citedby.uri?partnerID=HzOxMe3b&scp=33750589187&origin=inward'}],
                     'prism:url': 'https://api.elsevier.com/content/abstract/scopus_id/33750589187',
                     'dc:identifier': 'SCOPUS_ID:33750589187',
                     'eid': '2-s2.0-33750589187',
                     'dc:title': 'Histologic evaluation of implants following flapless and flapped surgery: A study in canines',
                     'dc:creator': 'Becker W.',
                     'prism:publicationName': 'Journal of Periodontology',
                     'prism:issn': '00223492',
                     'prism:volume': '77',
                     'prism:issueIdentifier': '10',
                     'prism:pageRange': '1717-1722',
                     'prism:coverDate': '2006-10-01',
                     'prism:coverDisplayDate': 'October 2006',
                     'prism:doi': '10.1902/jop.2006.060090',
                     'dc:description': 'Background: Flapless surgery requires penetration of the alveolar mucosa and bone without reflection of mucoperiosteal flaps. Do these techniques force gingival tissue or foreign materials into osteotomies? If so, do such tissues or materials interfere with osseointegration? A proof-of-principle study using a canine model attempted to answer these questions. Methods: Five young adult Hound Labrador mongrel dogs received implants with a moderately roughened surface by anodic oxidation using flapless or conventional one-stage (control) surgery in contralateral jaw quadrants. The implants were placed into the osteotomies, and the international stability quotient (ISQ) was recorded using resonance frequency analysis. These measurements were repeated following a 3-month healing interval when the animals were euthanized, and implants and surrounding tissues were retrieved and processed for histologic analysis. Results: The implants were stable upon insertion and demonstrated increased stability at 3 months without significant differences between surgical protocols. The histologic evaluation showed high bone-implant contact (flapless surgery: 54.7% ± 8.4%; control: 52.2% ± 13.0%; P >0.05) without evidence of gingival tissue or foreign body inclusions. There were no significant differences in marginal bone levels between the surgical protocols. Post-insertion and at 3 months, ISQ values depended on the amount of torque delivered. Immediately post-insertion, for every 1-unit increase in torque value, the ISQ increased by 0.3 (95% confidence interval: 0.1 to 0.4; P = 0.0043). Three months postoperatively, for every one-unit increase in torque the ISQ value decreased 0.2 (95% confidence interval: -0.4 to -0.1; P = 0.00 12). The effect of torque on ISQ values was independent of treatment effects and remained significant after adjustment for treatment. Conclusions: The results suggest that implants placed without flap reflection remain stable and exhibit clinically relevant osseointegration similar to when implants are placed with flapped procedures. Greater torque at implant placement resulted in less implant stability at 3 months.',
                     'citedby-count': '56',
                     'affiliation': [{'@_fa': 'true',
                       'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60024399',
                       'afid': '60024399',
                       'affilname': 'Medical College of Georgia',
                       'affiliation-city': 'Augusta',
                       'affiliation-country': 'United States'},
                      {'@_fa': 'true',
                       'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60019009',
                       'afid': '60019009',
                       'affilname': 'University of Southern California, School of Dentistry',
                       'affiliation-city': 'Los Angeles',
                       'affiliation-country': 'United States'},
                      {'@_fa': 'true',
                       'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60016437',
                       'afid': '60016437',
                       'affilname': 'Göteborgs Universitet',
                       'affiliation-city': 'Gothenburg',
                       'affiliation-country': 'Sweden'},
                      {'@_fa': 'true',
                       'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60015481',
                       'afid': '60015481',
                       'affilname': 'University of Washington, Seattle',
                       'affiliation-city': 'Seattle',
                       'affiliation-country': 'United States'},
                      {'@_fa': 'true',
                       'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60007903',
                       'afid': '60007903',
                       'affilname': 'Hebrew University of Jerusalem',
                       'affiliation-city': 'Jerusalem',
                       'affiliation-country': 'Israel'}],
                     'pubmed-id': '17032115',
                     'prism:aggregationType': 'Journal',
                     'subtype': 'ar',
                     'subtypeDescription': 'Article',
                     'author-count': {'@limit': '100', '$': '7'},
                     'author': [{'@_fa': 'true',
                       '@seq': '1',
                       'author-url': 'https://api.elsevier.com/content/author/author_id/55547124100',
                       'authid': '55547124100',
                       'authname': 'Becker W.',
                       'surname': 'Becker',
                       'given-name': 'William',
                       'initials': 'W.',
                       'afid': [{'@_fa': 'true', '$': '60019009'},
                        {'@_fa': 'true', '$': '60015481'}]},
                      {'@_fa': 'true',
                       '@seq': '2',
                       'author-url': 'https://api.elsevier.com/content/author/author_id/7007040311',
                       'authid': '7007040311',
                       'authname': 'Wikesjö U.',
                       'surname': 'Wikesjö',
                       'given-name': 'Ulf M.E.',
                       'initials': 'U.M.E.',
                       'afid': [{'@_fa': 'true', '$': '60024399'}]},
                      {'@_fa': 'true',
                       '@seq': '3',
                       'author-url': 'https://api.elsevier.com/content/author/author_id/7005071364',
                       'authid': '7005071364',
                       'authname': 'Sennerby L.',
                       'surname': 'Sennerby',
                       'given-name': 'Lars',
                       'initials': 'L.',
                       'afid': [{'@_fa': 'true', '$': '60016437'}]},
                      {'@_fa': 'true',
                       '@seq': '4',
                       'author-url': 'https://api.elsevier.com/content/author/author_id/6507418137',
                       'authid': '6507418137',
                       'authname': 'Qahash M.',
                       'surname': 'Qahash',
                       'given-name': 'Mohammed',
                       'initials': 'M.',
                       'afid': [{'@_fa': 'true', '$': '60024399'}]},
                      {'@_fa': 'true',
                       '@seq': '5',
                       'author-url': 'https://api.elsevier.com/content/author/author_id/7006267759',
                       'authid': '7006267759',
                       'authname': 'Hujoel P.',
                       'surname': 'Hujoel',
                       'given-name': 'Philippe',
                       'initials': 'P.',
                       'afid': [{'@_fa': 'true', '$': '60007903'}]},
                      {'@_fa': 'true',
                       '@seq': '6',
                       'author-url': 'https://api.elsevier.com/content/author/author_id/7403014169',
                       'authid': '7403014169',
                       'authname': 'Goldstein M.',
                       'surname': 'Goldstein',
                       'given-name': 'Moshe',
                       'initials': 'M.'},
                      {'@_fa': 'true',
                       '@seq': '7',
                       'author-url': 'https://api.elsevier.com/content/author/author_id/15052667100',
                       'authid': '15052667100',
                       'authname': 'Turkyilmaz I.',
                       'surname': 'Turkyilmaz',
                       'given-name': 'Ilser',
                       'initials': 'I.',
                       'afid': [{'@_fa': 'true', '$': '60016437'}]}],
                     'authkeywords': 'Dental implants | Histology | Osseointegration | Surgery',
                     'source-id': '26173',
                     'fund-no': 'undefined',
                     'openaccess': '0',
                     'openaccessFlag': False,
                     'struct:title': [[('Histologic', 'NNP'),
                       ('evaluation', 'NN'),
                       ('of', 'IN'),
                       ('implants', 'NNS'),
                       ('following', 'VBG'),
                       ('flapless', 'NN'),
                       ('and', 'CC'),
                       ('flapped', 'VBD'),
                       ('surgery', 'NN'),
                       (':', ':'),
                       ('A', 'DT'),
                       ('study', 'NN'),
                       ('in', 'IN'),
                       ('canines', 'NNS')]]}
        self.assertEqual(next(self.corp.docs()), target)

    def test_title_raw(self):
        target = 'Histologic evaluation of implants following flapless and flapped surgery: A study in canines'
        self.assertEqual(next(self.corp.title_raw()), target)

    def test_abstracts(self):
        target = 'Background: Flapless surgery requires penetration of the alveolar mucosa and bone without reflection of mucoperiosteal flaps. Do these techniques force gingival tissue or foreign materials into osteotomies? If so, do such tissues or materials interfere with osseointegration? A proof-of-principle study using a canine model attempted to answer these questions. Methods: Five young adult Hound Labrador mongrel dogs received implants with a moderately roughened surface by anodic oxidation using flapless or conventional one-stage (control) surgery in contralateral jaw quadrants. The implants were placed into the osteotomies, and the international stability quotient (ISQ) was recorded using resonance frequency analysis. These measurements were repeated following a 3-month healing interval when the animals were euthanized, and implants and surrounding tissues were retrieved and processed for histologic analysis. Results: The implants were stable upon insertion and demonstrated increased stability at 3 months without significant differences between surgical protocols. The histologic evaluation showed high bone-implant contact (flapless surgery: 54.7% ± 8.4%; control: 52.2% ± 13.0%; P >0.05) without evidence of gingival tissue or foreign body inclusions. There were no significant differences in marginal bone levels between the surgical protocols. Post-insertion and at 3 months, ISQ values depended on the amount of torque delivered. Immediately post-insertion, for every 1-unit increase in torque value, the ISQ increased by 0.3 (95% confidence interval: 0.1 to 0.4; P = 0.0043). Three months postoperatively, for every one-unit increase in torque the ISQ value decreased 0.2 (95% confidence interval: -0.4 to -0.1; P = 0.00 12). The effect of torque on ISQ values was independent of treatment effects and remained significant after adjustment for treatment. Conclusions: The results suggest that implants placed without flap reflection remain stable and exhibit clinically relevant osseointegration similar to when implants are placed with flapped procedures. Greater torque at implant placement resulted in less implant stability at 3 months.'
        self.assertEqual(next(self.corp.abstracts()), target)

    def test_doc_ids(self):
        target = 'https://api.elsevier.com/content/abstract/scopus_id/33750589187'
        self.assertEqual(next(self.corp.doc_ids()), target)
        target = 'SCOPUS_ID:33750589187'
        self.assertEqual(next(self.corp.doc_ids(form='dc:identifier')), target)
        target = '2-s2.0-33750589187'
        self.assertEqual(next(self.corp.doc_ids(form='eid')), target)
        target = ''
        self.assertEqual(next(self.corp.doc_ids(form='prism:isbn')), target)
        target = '10.1902/jop.2006.060090'
        self.assertEqual(next(self.corp.doc_ids(form='prism:doi')), target)
        target = ''
        self.assertEqual(next(self.corp.doc_ids(form='article-number')), target)

    def test_publication(self):
        target = 'Journal of Periodontology'
        self.assertEqual(next(self.corp.publication()), target)
        target = 'Article'
        self.assertEqual(next(self.corp.publication(form='subtypeDescription')),
                         target)
        target = 'Journal'
        self.assertEqual(next(
            self.corp.publication(form='prism:aggregationType')), target)
        target = 'ar'
        self.assertEqual(next(self.corp.publication(form='subtype')), target)

    def test_pub_date(self):
        target = datetime(2006, 10, 1, 0, 0)
        self.assertEqual(next(self.corp.pub_date()), target)
        target = 2006
        self.assertEqual(next(self.corp.pub_date(form='year')), target)

    def test_author_data(self):
        target = [{'@_fa': 'true',
                      '@seq': '1',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/55547124100',
                      'authid': '55547124100',
                      'authname': 'Becker W.',
                      'surname': 'Becker',
                      'given-name': 'William',
                      'initials': 'W.',
                      'afid': [{'@_fa': 'true', '$': '60019009'},
                       {'@_fa': 'true', '$': '60015481'}]},
                     {'@_fa': 'true',
                      '@seq': '2',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/7007040311',
                      'authid': '7007040311',
                      'authname': 'Wikesjö U.',
                      'surname': 'Wikesjö',
                      'given-name': 'Ulf M.E.',
                      'initials': 'U.M.E.',
                      'afid': [{'@_fa': 'true', '$': '60024399'}]},
                     {'@_fa': 'true',
                      '@seq': '3',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/7005071364',
                      'authid': '7005071364',
                      'authname': 'Sennerby L.',
                      'surname': 'Sennerby',
                      'given-name': 'Lars',
                      'initials': 'L.',
                      'afid': [{'@_fa': 'true', '$': '60016437'}]},
                     {'@_fa': 'true',
                      '@seq': '4',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/6507418137',
                      'authid': '6507418137',
                      'authname': 'Qahash M.',
                      'surname': 'Qahash',
                      'given-name': 'Mohammed',
                      'initials': 'M.',
                      'afid': [{'@_fa': 'true', '$': '60024399'}]},
                     {'@_fa': 'true',
                      '@seq': '5',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/7006267759',
                      'authid': '7006267759',
                      'authname': 'Hujoel P.',
                      'surname': 'Hujoel',
                      'given-name': 'Philippe',
                      'initials': 'P.',
                      'afid': [{'@_fa': 'true', '$': '60007903'}]},
                     {'@_fa': 'true',
                      '@seq': '6',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/7403014169',
                      'authid': '7403014169',
                      'authname': 'Goldstein M.',
                      'surname': 'Goldstein',
                      'given-name': 'Moshe',
                      'initials': 'M.'},
                     {'@_fa': 'true',
                      '@seq': '7',
                      'author-url': 'https://api.elsevier.com/content/author/author_id/15052667100',
                      'authid': '15052667100',
                      'authname': 'Turkyilmaz I.',
                      'surname': 'Turkyilmaz',
                      'given-name': 'Ilser',
                      'initials': 'I.',
                      'afid': [{'@_fa': 'true', '$': '60016437'}]}]
        self.assertEqual(next(self.corp.author_data()), target)

    def test_author_count(self):
        target = 7
        self.assertEqual(next(self.corp.author_count()), target)

    def test_author_name(self):
        target = 'Becker W.'
        self.assertEqual(next(self.corp.author_name()), target)
        target = 'https://api.elsevier.com/content/author/author_id/55547124100'
        self.assertEqual(next(self.corp.author_name(form='author-url')),
                         target)
        target = '55547124100'
        self.assertEqual(next(self.corp.author_name(form='authid')), target)
        target = 'Becker'
        self.assertEqual(next(self.corp.author_name(form='surname')), target)
        target = 'William'
        self.assertEqual(next(self.corp.author_name(form='given-name')),
                         target)
        target = 'W.'
        self.assertEqual(next(self.corp.author_name(form='initials')), target)

    def test_author_list(self):
        target = ['Becker W.',
                     'Wikesjö U.',
                     'Sennerby L.',
                     'Qahash M.',
                     'Hujoel P.',
                     'Goldstein M.',
                     'Turkyilmaz I.']
        self.assertEqual(next(self.corp.author_list()), target)
        target = ['https://api.elsevier.com/content/author/author_id/55547124100',
                     'https://api.elsevier.com/content/author/author_id/7007040311',
                     'https://api.elsevier.com/content/author/author_id/7005071364',
                     'https://api.elsevier.com/content/author/author_id/6507418137',
                     'https://api.elsevier.com/content/author/author_id/7006267759',
                     'https://api.elsevier.com/content/author/author_id/7403014169',
                     'https://api.elsevier.com/content/author/author_id/15052667100']
        self.assertEqual(next(self.corp.author_list(form='author-url')),
                         target)
        target = ['55547124100',
                     '7007040311',
                     '7005071364',
                     '6507418137',
                     '7006267759',
                     '7403014169',
                     '15052667100']
        self.assertEqual(next(self.corp.author_list(form='authid')), target)
        target = ['Becker',
                     'Wikesjö',
                     'Sennerby',
                     'Qahash',
                     'Hujoel',
                     'Goldstein',
                     'Turkyilmaz']
        self.assertEqual(next(self.corp.author_list(form='surname')), target)
        target = ['William', 'Ulf M.E.', 'Lars', 'Mohammed', 'Philippe',
                  'Moshe', 'Ilser']
        self.assertEqual(next(self.corp.author_list(form='given-name')),
                         target)
        target = ['W.', 'U.M.E.', 'L.', 'M.', 'P.', 'M.', 'I.']
        self.assertEqual(next(self.corp.author_list(form='initials')), target)

    def test_author_keyword_list(self):
        target = ['Dental implants', 'Histology', 'Osseointegration', 'Surgery']
        self.assertEqual(next(self.corp.author_keyword_list()), target)

    def test_author_keyword(self):
        target = 'Dental implants'
        self.assertEqual(next(self.corp.author_keyword()), target)

    def test_doc_volume(self):
        target = '77'
        self.assertEqual(next(self.corp.doc_volume()), target)

    def test_doc_page_range(self):
        target = (1717, 1722)
        self.assertEqual(next(self.corp.doc_page_range()), target)

    def test_doc_citation_number(self):
        target = 56
        self.assertEqual(next(self.corp.doc_citation_number()), target)

    def test_affiliation_list(self):
        target =   [{'@_fa': 'true',
                     'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60024399',
                      'afid': '60024399',
                      'affilname': 'Medical College of Georgia',
                      'affiliation-city': 'Augusta',
                      'affiliation-country': 'United States'},
                     {'@_fa': 'true',
                      'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60019009',
                      'afid': '60019009',
                      'affilname': 'University of Southern California, School of Dentistry',
                      'affiliation-city': 'Los Angeles',
                      'affiliation-country': 'United States'},
                     {'@_fa': 'true',
                      'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60016437',
                      'afid': '60016437',
                      'affilname': 'Göteborgs Universitet',
                      'affiliation-city': 'Gothenburg',
                      'affiliation-country': 'Sweden'},
                     {'@_fa': 'true',
                      'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60015481',
                      'afid': '60015481',
                      'affilname': 'University of Washington, Seattle',
                      'affiliation-city': 'Seattle',
                      'affiliation-country': 'United States'},
                     {'@_fa': 'true',
                      'affiliation-url': 'https://api.elsevier.com/content/affiliation/affiliation_id/60007903',
                      'afid': '60007903',
                      'affilname': 'Hebrew University of Jerusalem',
                      'affiliation-city': 'Jerusalem',
                      'affiliation-country': 'Israel'}]
        self.assertEqual(next(self.corp.affiliation_list()), target)


class TestScopusProcessedCorpusReader(TestCase):
    def setUp(self) -> None:
        self.corp = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")

    def test_title_sents(self):
        target = ('Histologic', 'NNP')
        result = next(self.corp.title_sents())[0]
        self.assertEqual(result, target)

    def test_title_tagged(self):
        target = ('Histologic', 'NNP')
        result = next(self.corp.title_tagged_word())
        self.assertEqual(result, target)

    def test_title_words(self):
        target = 'Histologic'
        result = next(self.corp.title_words())
        self.assertEqual(result, target)

    def test_describe(self):
        target = {'files': 56350,
                  'topics': 1,
                  'titles': 56350,
                  'words': 694742,
                  'vocab': 33065,
                  'lexdiv': 21.011401784364132,
                  'tpdoc': 1.0,
                  'wptit': 12.32905057675244}

        result = self.corp.describe()
        for metric in target:
            self.assertEqual(result[metric], target[metric])


class TestCorpuSubsetLoader(TestCase):
    def setUp(self) -> None:
        self.corp = Elsevier_Corpus_Reader.ScopusProcessedCorpusReader(
            "Corpus/Processed_corpus/")
        self.loader = Elsevier_Corpus_Reader.CorpuKfoldLoader(self.corp,
                                                              n_folds=12,
                                                              shuffle=False)

    def test_fileids(self):
        target = 4696
        result = len(next(self.loader.fileids(test=True)))
        self.assertEqual(result, target)
        target = 51654
        result = len(next(self.loader.fileids(train=True)))
        self.assertEqual(result, target)


