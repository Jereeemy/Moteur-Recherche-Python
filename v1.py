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
        self.type = self.getType()  

    def getType(self):
        return "Document" 

    def __str__(self):
        return f"{self.title} by {self.author} on {self.date} ({self.type})"

# Classe pour les documents Reddit
class RedditDocument(Document):
    def __init__(self, title, author, date, num_comments, text=""):
        super().__init__(title, author, date, text)  # Appel au constructeur parent
        self.num_comments = num_comments

    def getType(self):
        return "Reddit"  # Type spécifique pour Reddit

    def __str__(self):
        return super().__str__() + f", Comments: {self.num_comments}"

# Classe pour les documents Arxiv
class ArxivDocument(Document):
    def __init__(self, title, authors, date, text=""):
        super().__init__(title, authors, date, text)  # Appel au constructeur parent
        self.authors = authors  

    def getType(self):
        return "Arxiv"  # Type spécifique pour Arxiv

    def __str__(self):
        authors_str = ", ".join(self.authors)
        return super().__str__() + f", Authors: {authors_str}"

# Classe pour gérer un corpus de documents (Singleton)
class Corpus:
    _instance = None  # Instance unique pour le singleton

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Corpus, cls).__new__(cls)
            cls._instance.documents = []  # Initialisation de la liste
        return cls._instance  

    def add_document(self, document):
        self.documents.append(document)  # Ajout d'un document

    def display_documents(self):
        for doc in self.documents:
            print(doc)  # Affichage de chaque document

# Fonction de collecte des données de Reddit
def collect_reddit_data():
    reddit = praw.Reddit(client_id='tEQSvErg_mor-oAPoqEjlg',client_secret='FbQthefsNh8wiLCYlkAjeFx7L1VLeA',user_agent='WebScrapping')
    hot_posts = reddit.subreddit('Coronavirus')
    documents = []
    for idx, post in enumerate(hot_posts.hot(limit=100)):
        combined_text = f"{post.title}. {post.selftext}".replace("\n", " ")
        doc = RedditDocument(
            title=post.title,
            author=str(post.author) if post.author else "Inconnu",
            date=str(datetime.datetime.fromtimestamp(post.created_utc)),
            num_comments=post.num_comments,
            text=combined_text
        )
        documents.append(doc)
    return documents

# Fonction de collecte des données d'Arxiv
def collect_arxiv_data():
    query = "covid"
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=100'
    time.sleep(1)  # Respecter les limites de taux

    with urllib.request.urlopen(url) as response:
        data = response.read()

    parsed_data = xmltodict.parse(data)
    documents = []
    for idx, entry in enumerate(parsed_data['feed']['entry']):
        auteur = entry['author'][0].get('name', "Inconnu") if 'author' in entry and isinstance(entry['author'], list) and entry['author'] else "Inconnu"
        doc = ArxivDocument(
            title=entry['title'],
            authors=[auteur],
            date=entry['published'],
            text=entry['summary'].replace("\n", " ")
        )
        documents.append(doc)
    return documents

# Exemple d'utilisation
if __name__ == "__main__":
    corpus = Corpus()  # Création ou récupération de l'instance unique de Corpus

    # Collecte des données
    reddit_documents = collect_reddit_data()
    arxiv_documents = collect_arxiv_data()

    # Ajout des documents au corpus
    for doc in reddit_documents + arxiv_documents:
        corpus.add_document(doc)

    # Affichage des documents dans le corpus
    corpus.display_documents()

    # Sauvegarde du corpus dans un fichier pickle
    with open("corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)

    # Lecture du corpus depuis un fichier pickle
    with open("corpus.pkl", "rb") as f:
        loaded_corpus = pickle.load(f)
        loaded_corpus.display_documents()  # Affichage des documents chargés
