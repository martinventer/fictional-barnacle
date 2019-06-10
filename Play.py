
from Ingestor import Elsevier_Ingestor

def main():
    search_terms = ['soft robot']
    dates = (2011, 2012)


    builder = Elsevier_Ingestor.ElsevierIngestionEngine(search_terms=search_terms,
                                                    file_path="corpus/",
                                                    dates=dates,
                                                    home=False,
                                                    batch_size=25)

    builder.build_corpus()


if __name__ == '__main__':
    main()