import unittest
from v1 import Document, RedditDocument, ArxivDocument, Corpus

# Teste la création d'un document de type Document
class TestDocument(unittest.TestCase):
    def test_document_creation(self):
        doc = Document("Titre Test", "Nom de l'Auteur", "2024-01-01")
        print(f"Création du document : {doc}")  
        self.assertEqual(doc.title, "Titre Test")
        self.assertEqual(doc.author, "Nom de l'Auteur")
        self.assertEqual(doc.date, "2024-01-01")
        self.assertEqual(doc.getType(), "Document")

# Teste la création d'un document de type RedditDocument et vérifie le nombre de commentaires
    def test_reddit_document(self):
        doc = RedditDocument("Titre Reddit", "Redditor", "2024-01-01", 5)
        print(f"Création du document Reddit : {doc}")  
        self.assertEqual(doc.getType(), "Reddit")
        self.assertEqual(doc.num_comments, 5)

# Teste la création d'un document de type ArxivDocument et vérifie les auteurs
    def test_arxiv_document(self):
        doc = ArxivDocument("Titre Arxiv", ["Auteur1", "Auteur2"], "2024-01-01")
        print(f"Création du document Arxiv : {doc}")  
        self.assertEqual(doc.getType(), "Arxiv")
        self.assertIn("Auteur1", str(doc))

# Teste l'implémentation du design pattern Singleton pour la classe Corpus
    def test_corpus_singleton(self):
        corpus1 = Corpus()
        corpus2 = Corpus()
        self.assertIs(corpus1, corpus2)
        doc = Document("Document Test", "Auteur", "2024-01-01")
        corpus1.add_document(doc)
        print(f"Document ajouté au corpus : {doc}")  
        self.assertIn(doc, corpus2.documents)

if __name__ == "__main__":
    unittest.main(verbosity=2)
