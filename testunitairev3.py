import unittest
from datetime import datetime
from v3 import Document, RedditDocument, ArxivDocument, Corpus, fetch_arxiv_data, fetch_reddit_data, get_keyword_suggestions


class TestCorpus(unittest.TestCase):
    # Teste la construction du vocabulaire pour vérifier les fréquences des mots dans le corpus.
    def test_construire_vocabulaire(self):
        corpus = Corpus()
        doc1 = Document("Title1", "Author1", "2023-12-01", "word word another")
        doc2 = Document("Title2", "Author2", "2023-12-02", "word new")
        corpus.add_document(doc1)
        corpus.add_document(doc2)
        vocab = corpus.construire_vocabulaire()
        print(f"Test de la construction du vocabulaire : {vocab}")
        self.assertEqual(vocab["word"], 3)
        self.assertEqual(vocab["another"], 1)
        self.assertEqual(vocab["new"], 1)


class TestFetchFunctions(unittest.TestCase):
    # Teste la récupération des données depuis Reddit pour vérifier l'accès et le format des posts.
    def test_fetch_reddit_data(self):
        try:
            documents = fetch_reddit_data("python", limit=1)
            print(f"Test de récupération des données Reddit : {documents}")
            self.assertTrue(len(documents) > 0)
        except Exception:
            self.skipTest("fetch_reddit_data nécessite des accès API valides pour tester")

    # Teste la récupération des données depuis Arxiv pour s'assurer de la validité des documents récupérés.
    def test_fetch_arxiv_data(self):
        documents = fetch_arxiv_data("machine+learning", max_results=1)
        print(f"Test de récupération des données Arxiv : {documents}")
        self.assertTrue(len(documents) > 0)
        self.assertTrue(all(isinstance(doc, ArxivDocument) for doc in documents))


class TestKeywordSuggestions(unittest.TestCase):
    # Teste la fonction de suggestion de mots-clés pour vérifier l'extraction des mots les plus fréquents dans le corpus.
    def test_get_keyword_suggestions(self):
        corpus = Corpus()
        doc1 = Document("Title1", "Author1", "2023-12-01", "machine learning AI")
        doc2 = Document("Title2", "Author2", "2023-12-02", "AI machine data")
        corpus.add_document(doc1)
        corpus.add_document(doc2)
        keywords = get_keyword_suggestions(corpus, top_n=2)
        print(f"Suggestions de mots-clés : {keywords}")
        self.assertIn("machine", keywords)
        self.assertIn("ai", keywords)


if __name__ == '__main__':
    unittest.main(verbosity=2)
