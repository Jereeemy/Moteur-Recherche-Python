import unittest
from datetime import datetime
from v3 import get_keyword_suggestions, fetch_arxiv_data, Corpus, fetch_reddit_data, Document, ArxivDocument, RedditDocument, load_corpus_from_reddit_and_arxiv, display_results_as_table

class TestSearchEngine(unittest.TestCase):

    # Teste la génération de suggestions de mots-clés à partir d'un corpus
    def test_get_keyword_suggestions(self):
        corpus = Corpus()
        corpus.add_document(Document("Python Programming", "John Doe", "2023-01-01", "Python is a great programming language"))
        corpus.add_document(Document("Data Science", "Jane Smith", "2023-01-02", "Data science uses Python and machine learning"))
        suggestions = get_keyword_suggestions(corpus, top_n=3)
        print(f"Suggestions de mots-clés obtenues : {suggestions}")
        self.assertEqual(len(suggestions), 3)
        self.assertIn("python", suggestions)
        self.assertIn("machine", suggestions)
        self.assertIn("learning", suggestions)

    # Vérifie la récupération et le format des données depuis arXiv
    def test_fetch_arxiv_data(self):
        results = fetch_arxiv_data("python", max_results=5)
        print(f"Nombre de documents arXiv récupérés : {len(results)}")
        print(f"Premier document arXiv : {results[0]}")
        self.assertLessEqual(len(results), 5)
        self.assertIsInstance(results[0], ArxivDocument)
        self.assertIsNotNone(results[0].title)
        self.assertIsNotNone(results[0].authors)
        self.assertIsInstance(results[0].date, datetime)
        self.assertIsNotNone(results[0].content)

    # Teste la récupération et le format des données depuis Reddit
    def test_fetch_reddit_data(self):
        results = fetch_reddit_data("python", limit=5)
        print(f"Nombre de documents Reddit récupérés : {len(results)}")
        print(f"Premier document Reddit : {results[0]}")
        self.assertLessEqual(len(results), 5)
        self.assertIsInstance(results[0], RedditDocument)
        self.assertIsNotNone(results[0].title)
        self.assertIsNotNone(results[0].author)
        self.assertIsInstance(results[0].date, datetime)
        self.assertIsNotNone(results[0].content)
        self.assertIsInstance(results[0].num_comments, int)

    # Vérifie le chargement du corpus à partir de Reddit et arXiv
    def test_load_corpus_from_reddit_and_arxiv(self):
        corpus = load_corpus_from_reddit_and_arxiv()
        print(f"Nombre total de documents dans le corpus : {len(corpus.documents)}")
        print(f"Nombre de documents Reddit : {sum(1 for doc in corpus.documents if isinstance(doc, RedditDocument))}")
        print(f"Nombre de documents arXiv : {sum(1 for doc in corpus.documents if isinstance(doc, ArxivDocument))}")
        self.assertIsNotNone(corpus)
        self.assertGreater(len(corpus.documents), 0)
        reddit_docs = [doc for doc in corpus.documents if isinstance(doc, RedditDocument)]
        arxiv_docs = [doc for doc in corpus.documents if isinstance(doc, ArxivDocument)]
        self.assertGreater(len(reddit_docs), 0)
        self.assertGreater(len(arxiv_docs), 0)

    # Teste l'affichage des résultats sous forme de tableau
    def test_display_results_as_table(self):
        corpus = load_corpus_from_reddit_and_arxiv()
        results = corpus.documents[:2]
        print("Affichage des résultats sous forme de tableau :")
        display_results_as_table(results)
        
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main(verbosity=2)
