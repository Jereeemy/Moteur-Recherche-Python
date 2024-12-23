import praw
import urllib.request
import xmltodict
import datetime
import pickle
import time

# Classe de base pour les documents
class Document:
    def __init__(self, title, author, date, text=""):
        self.title = title  
        self.author = author  
        self.date = date  
        self.text = text  
        self.type = self.getType()  # Déterminé dynamiquement par les classes filles

    def getType(self):
        return "Document"  # Type générique par défaut

    def __str__(self):
        return f"{self.title} by {self.author} on {self.date} ({self.type})"

# Classe pour les documents Reddit
class RedditDocument(Document):
    def __init__(self, title, author, date, num_comments, text=""):
        super().__init__(title, author, date, text)  # Hérite des attributs communs de Document
        self.num_comments = num_comments  # Nombre de commentaires sur le post

    def getType(self):
        return "Reddit"  # Spécifique aux documents Reddit

    def __str__(self):
        return super().__str__() + f", Comments: {self.num_comments}"

# Classe pour les documents Arxiv
class ArxivDocument(Document):
    def __init__(self, title, authors, date, text=""):
        super().__init__(title, authors, date, text)  # Hérite des attributs communs de Document
        self.authors = authors  # Liste des auteurs

    def getType(self):
        return "Arxiv"  # Spécifique aux documents Arxiv

    def __str__(self):
        authors_str = ", ".join(self.authors)  # Jointure des auteurs en une chaîne
        return super().__str__() + f", Authors: {authors_str}"

# Classe pour gérer un corpus de documents (Singleton)
class Corpus:
    _instance = None  # Instance unique pour le Singleton

    def __new__(cls):
        # Implémentation du Singleton pour assurer une seule instance
        if cls._instance is None:
            cls._instance = super(Corpus, cls).__new__(cls)
            cls._instance.documents = []  # Initialisation de la liste des documents
        return cls._instance  

    def add_document(self, document):
        self.documents.append(document)  # Ajout d'un document au corpus

    def display_documents(self):
        for doc in self.documents:
            print(doc)  # Affichage de chaque document

# Fonction de collecte des données de Reddit
def collect_reddit_data():
    # Configuration de l'API Reddit 
    reddit = praw.Reddit(client_id='tEQSvErg_mor-oAPoqEjlg', client_secret='FbQthefsNh8wiLCYlkAjeFx7L1VLeA', user_agent='WebScrapping')
    hot_posts = reddit.subreddit('Coronavirus')  # Sélectionne le subreddit 'Coronavirus'
    documents = []
    for idx, post in enumerate(hot_posts.hot(limit=100)):  # Récupère les 100 posts les plus populaires
        combined_text = f"{post.title}. {post.selftext}".replace("\n", " ")  # Combine le titre et le texte du post
        doc = RedditDocument(
            title=post.title,
            author=str(post.author) if post.author else "Inconnu",  
            date=str(datetime.datetime.fromtimestamp(post.created_utc)),  # Conversion de l'horodatage en date
            num_comments=post.num_comments,
            text=combined_text
        )
        documents.append(doc)
    return documents

# Fonction de collecte des données d'Arxiv
def collect_arxiv_data():
    query = "covid"  # Terme de recherche dans l'API Arxiv
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=100'  # URL de l'API
    time.sleep(1)  # Pause pour éviter un dépassement du taux de requêtes

    with urllib.request.urlopen(url) as response:
        data = response.read()  # Lecture des données renvoyées par l'API

    parsed_data = xmltodict.parse(data)  # Conversion des données XML en dictionnaire
    documents = []
    for idx, entry in enumerate(parsed_data['feed']['entry']):
        auteur = entry['author'][0].get('name', "Inconnu") if 'author' in entry and isinstance(entry['author'], list) and entry['author'] else "Inconnu"
        doc = ArxivDocument(
            title=entry['title'],
            authors=[auteur],
            date=entry['published'],  # Date de publication
            text=entry['summary'].replace("\n", " ")  # Résumé du document sans sauts de ligne
        )
        documents.append(doc)
    return documents

# Exemple d'utilisation
if __name__ == "__main__":
    corpus = Corpus()  # Récupération de l'instance unique du Corpus

    # Collecte des données depuis Reddit et Arxiv
    reddit_documents = collect_reddit_data()
    arxiv_documents = collect_arxiv_data()

    # Ajout des documents au corpus
    for doc in reddit_documents + arxiv_documents:
        corpus.add_document(doc)

    # Affichage des documents dans le corpus
    corpus.display_documents()

    # Sauvegarde du corpus dans un fichier pickle
    with open("corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)  # Sérialisation et sauvegarde

    # Lecture du corpus depuis un fichier pickle
    with open("corpus.pkl", "rb") as f:
        loaded_corpus = pickle.load(f)  # Chargement du corpus sérialisé
        loaded_corpus.display_documents()  # Vérification des documents chargés
