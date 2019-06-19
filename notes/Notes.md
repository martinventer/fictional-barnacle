# NOTES and USEFUL THINGS

## Science Direct Search STANDARD view 

### fields according to API Documentation
https://dev.elsevier.com/guides/ScienceDirectSearchViews.htm

| Field                 | Description                       | STANDARD |
|-----------------------|-----------------------------------|----------|
| link ref=self         | Content Full-Text Article API URI | X        |
| link ref=scidir       | ScienceDirect article page URI    | X        |
| prism:url             | Content Full-Text Article API URI | X        |
| dc:identifier         | DOI                               | X        |
| openaccess            | Open Access Status                | X        |
| dc:title              | Article Title                     | X        |
| prism:publicationName | Source title                      | X        |
| prism:volume          | Volume number(serial only)        | X        |
| prism:startingPage    | Start page                        | X        |
| prism:endingPage      | End page                          | X        |
| prism:coverDate       | Publication date (YYYY-MM-DD)     | X        |
| dc:creator            | First Author                      | X        |
| authors               | Full Author List                  | X        |
| prism:doi             | Document Object Identifier        | X        |
| pii                   | Publication Item Identifier       | X        |

### Example from existing download
* '@_fa': 'true',
* 'load-date': '2003-07-30T00:00:00Z',
* 'link': 
    * [{
        * '@_fa': 'true',
        * '@ref': 'self', 
        * '@href': 'https://api.elsevier.com/content/article/pii/0167713686900351'
    * }, 
    * {
        * '@_fa': 'true',
        * '@ref': 'scidir', 
        * '@href': 'https://www.sciencedirect.com/science/article/pii/0167713686900351?dgcid=api_sd_search-api-endpoint'
    * }], 
* 'dc:identifier': 'DOI:10.1016/0167-7136(86)90035-1', 
* 'prism:url': 'https://api.elsevier.com/content/article/pii/0167713686900351',
* 'dc:title': 'Knowledge resource tools for information access', 
* 'dc:creator': 'D. E. Walker', 
* 'prism:publicationName': 'Computer Compacts',
* 'prism:volume': '4', 
* 'prism:coverDate': '1986-10-31',
* 'prism:startingPage': '182', 
* 'prism:doi': '10.1016/0167-7136(86)90035-1', 
* 'openaccess': False, 
* 'pii': '0167713686900351',
* 'authors': {'author': 'D. E. Walker'}}

### Included in Corpus Reader
* docs() - full content of document
* title_raw() - 'dc:title'
* title_words() - words in tokenized words in title
* doc_ids() - 'dc:identifier'
* publication() - 'prism:publicationName'
* pub_date() - 'prism:coverDate'
* pub_type() - 'subtypeDescription'
* author_list() - 'author'
* author_keywords() - 'authkeywords'
* doc_url() - 'prism:url'
* doc_volume() - 'prism:volume'
* doc_first_page() - 'prism:startingPage'
* doc_doi() - 'prism:doi'
* doc_pii() - 'pii'
* () - 



