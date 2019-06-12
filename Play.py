
from Ingestor import Elsivier_Ingestor
from CorpusReader import Elsevier_Corpus_Reader
from CorpusProcessingTools import Elsivier_Corpus_Pre_Processor
from datetime import datetime

from collections import Counter

corp = Elsevier_Corpus_Reader.PickledCorpusReader(
    "Corpus/Processed_corpus/")

for doc in corp.docs():
    try:
        doc["authors"]
    except KeyError:
        print(doc)
        break


ter = {"thing": 1}
ter['tt'] = 2
ter