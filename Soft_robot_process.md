# step 1: download the raw corpus from elsivier
* search the Scopus database for the term 'soft robot
* search over the date range 1950 to 2021
* Downloaded on date 26 August 2019
# step 2: refactor the corpus
* The corpus is downloaded as collections containing all publications
  within a given year. This step splits these collections into
  individual documents that can be accessed independently.
# step 3: Pre process the corpus to clean and format the data.
* tokenize text into format where one document returns
    [list of paragraphs
        [list of sentences
            [list of tagged tokens
                (token, tag)]]]
* tokenize additional text fields such as author, city, journal names
* add the file path each document

# step 4: Preliminary exploration
## Details of the corpus
* number of files: 58711
* descriptions: 
* word count: 11344916 (words)
* Vocabulary: 99243 (tokens)
* lexical diversity: 334.92 (number of words per token)
* Words per description: 193.23
* titles:
* word count: 725721 (words)
* Vocabulary: 33873 (tokens)
* lexical diversity: 21.42 (number of words per token)
* Words per title: 12.36
    