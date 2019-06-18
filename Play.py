from Ingestor import Elsivier_Ingestor
from CorpusProcessingTools import Elsivier_Corpus_Pre_Processor
from CorpusReader import Elsevier_Corpus_Reader
from CorpusProcessingTools import Plotting_Tools, Author_Networks


# AN = Author_Networks.AuthorNetworks("Corpus/Processed_corpus/")
# AN.co_author_network_bokeh(categories='soft robot/2000')


corp = Elsevier_Corpus_Reader.ScopusPickledCorpusReader(
    "Corpus/Processed_corpus/")

gen = corp.docs(categories='soft robot/2000')
next(gen)