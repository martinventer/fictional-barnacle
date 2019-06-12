
from Ingestor import Elsevier_Ingestor
from CorpusReader import Elsevier_Corpus_Reader
from CorpusProcessingTools import Elsivier_Corpus_Pre_Processor
from datetime import datetime

from collections import Counter

corp = Elsevier_Corpus_Reader.PickledCorpusReader(
        "Corpus/Processed_corpus/")

dates = list(corp.pub_date())


dcounts = Counter(d.year for d in dates)
for d, count in dcounts.items():
    print('The total defects for date {} is {}'.format(d, count))

test = Counter(corp.pub_date(form='year'))

