
from Ingestor import Elsivier_Ingestor
from CorpusReader import Elsevier_Corpus_Reader
from CorpusProcessingTools import Elsivier_Corpus_Pre_Processor
from datetime import datetime

from collections import Counter

corp = Elsevier_Corpus_Reader.ScopusPickledCorpusReader(
    "Corpus/Processed_corpus/")

# gen = corp.affiliation_list(categories='soft robot/2019')
# for i in range(10): print(next(gen))
#
# gen = corp.pub_date(categories='soft robot/2019', form='year')
# for i in range(10): print(next(gen))


gen = corp.docs(categories='soft robot/2019')
next(gen)