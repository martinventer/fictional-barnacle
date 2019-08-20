# NOTES and USEFUL THINGS

# never update the master without checking first


## exporting current package list and conda environemnt
conda env export > notes/environment.yaml

pip freeze > notes/requirements.txt

conda list -e > requirements.txt


# Filters

"""
list of methods for filetering
[
====affiliation
'affiliation_city_l',
'affiliation_city_s',
'affiliation_country_l',
'affiliation_country_s',
'affiliation_id_l',
'affiliation_id_s',
'affiliation_l',
'affiliation_name_l',
'affiliation_name_s',
'affiliation_url_l',
'affiliation_url_s',
====author
'author_data_id_l',
'author_data_id_s',
'author_data_initial_l',
'author_data_initial_s',
'author_data_l',
'author_data_name_full_l',
'author_data_name_full_s',
'author_data_name_given_l',
'author_data_name_given_s',
'author_data_name_surname_l',
'author_data_name_surname_s',
'author_data_url_l',
'author_data_url_s',
====description
'description_ngrams',
'description_tagged',
'description_tagged_sents',
'description_tagged_word',
'description_word',
>>>>'docs',
====identifier
'identifier_doi',
'identifier_electronic',
'identifier_issn',
'identifier_pubmed',
'identifier_scopus',
'identifier_source',
====keywords
'keywords_l',
'keywords_phrase',
'keywords_s',
'keywords_string',
====publication
'publication_date',
!!!!'publication_issue',
'publication_name',
!!!!'publication_pages',
'publication_subtype',
'publication_type',
!!!!'publication_volume',
'publication_year',
====stats
'stat_num_authors',
'stat_num_citations',
====title
'title_ngrams',
'title_tagged',
'title_tagged_sents',
'title_tagged_word',
'title_word',]

"""
