import unittest
from v2 import Document, RedditDocument, ArxivDocument, Corpus

class TestDocument(unittest.TestCase):

    # Teste la méthode de recherche de concordances dans un corpus de documents.
    # Vérifie que des résultats sont retournés pour une recherche de concordance avec le mot "covid".
    def test_concorde(self):
        corpus = Corpus()
        doc = Document("Test Document", "Auteur", "2024-01-01")
        corpus.add_document(doc)

        concordances = corpus.concorde("COVID", context_size=10)
        print(f"Concordances : {concordances}")
        self.assertGreater(len(concordances), 0, "Les concordances devraient retourner des résultats.")

    # Teste la méthode stats du corpus, qui calcule et affiche les statistiques sur les mots 
    # principalement la fréquence des mots du corpus.
    def test_stats(self):
        corpus = Corpus()
        doc1 = Document("COVID-19 Vaccine", "Auteur Test", "2024-01-01")
        doc2 = Document("COVID-19 Treatment", "Auteur Test", "2024-01-02")
        corpus.add_document(doc1)
        corpus.add_document(doc2)

        print("Statistiques sur les mots du corpus :")
        corpus.stats(n=10)  # Affichage des 10 mots les plus fréquents

    # Teste la recherche par similarité cosinus dans le corpus, vérifiant que la méthode
    # de recherche fonctionne correctement pour un mot-clé donné.
    def test_search_cosine(self):
        corpus = Corpus()
        doc1 = Document("COVID-19 impact", "Auteur Test", "2024-01-01")
        doc2 = Document("Health and COVID", "Auteur Test", "2024-01-02")
        corpus.add_document(doc1)
        corpus.add_document(doc2)

        query = "COVID" 
        print("Test de la recherche par similarité cosinus pour le mot-clé 'COVID' :")
        corpus.search_cosine(query)

    # Teste la méthode de recherche par mots-clés dans le corpus, vérifiant que la méthode
    # retourne correctement les résultats pour un mot-clé donné dans plusieurs documents.
    def test_search(self):
        corpus = Corpus()
        doc1 = Document("Document Test 1", "Auteur Test", "2024-01-01")
        doc2 = Document("Document Test 2", "Auteur Test", "2024-01-02")
        corpus.add_document(doc1)
        corpus.add_document(doc2)

        resultats = corpus.search("COVID")
        print(f"Résultats de la recherche : {resultats}")
        self.assertGreater(len(resultats), 0, "La recherche devrait retourner des résultats.")

# Lancement des tests
if __name__ == "__main__":
    unittest.main(verbosity=2)
