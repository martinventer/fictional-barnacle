from Ingestor import Elsivier_Ingestor
from CorpusProcessingTools import Elsivier_Corpus_Pre_Processor
from CorpusReader import Elsevier_Corpus_Reader
from CorpusProcessingTools import Plotting_Tools, Author_Networks

# step 1: download the raw corpus from elsivier
# download_corpus()

# step 2: reformat the corpus for faster manipulation
# preprocess_corpus()

# step 3: reformat the corpus for faster manipulation
# preprocess_corpus()

# step 4: load the corpus reader
# corp = Elsevier_Corpus_Reader.ScopusPickledCorpusReader(
#     "Corpus/Processed_corpus/")

# step 5: plot author connectivity
# AN = Author_Networks.AuthorNetworks("Corpus/Processed_corpus/")
# AN.plot_co_author_network(categories='soft robot/2000')