#!pip install praw
import praw
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Classe de base pour les documents
class Document:
    def __init__(self, title, author, date, content=None):
        self.title = title
        self.author = author
        self.date = datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date
        self.content = content if content else ""

    def __repr__(self):
        return f"Document({self.title}, {self.author}, {self.date}, {self.content[:100]}...)"

# Classe pour les documents Reddit
class RedditDocument(Document):
    def __init__(self, title, author, date, num_comments, content=None):
        super().__init__(title, author, date, content)
        self.num_comments = num_comments

    def getType(self):
        return "Reddit"

    def __str__(self):
        return super().__str__() + f", Comments: {self.num_comments}"

# Classe pour les documents Arxiv
class ArxivDocument(Document):
    def __init__(self, title, authors, date, content=None):
        super().__init__(title, authors, date, content)
        self.authors = authors

    def getType(self):
        return "Arxiv"

    def __str__(self):
        authors_str = ", ".join(self.authors)
        return super().__str__() + f", Authors: {authors_str}"

# Classe Corpus
class Corpus:
    def __init__(self):
        self.documents = []

    def add_document(self, document):
        self.documents.append(document)

    def search(self, keyword):
        return [doc for doc in self.documents if keyword.lower() in doc.content.lower()]

    def concorde(self, keyword):
        return [doc.content for doc in self.documents if keyword.lower() in doc.content.lower()]

    def construire_vocabulaire(self):
        vocab = {}
        for doc in self.documents:
            for word in doc.content.split():
                vocab[word] = vocab.get(word, 0) + 1
        return vocab

# Récupérer les données de Reddit
def fetch_reddit_data(subreddit_name, limit=100):
    reddit = praw.Reddit(client_id='tEQSvErg_mor-oAPoqEjlg',
                         client_secret='FbQthefsNh8wiLCYlkAjeFx7L1VLeA',
                         user_agent='WebScrapping')
    subreddit = reddit.subreddit(subreddit_name)
    posts = subreddit.new(limit=limit)
    
    documents = []
    for post in posts:
        title = post.title
        author = post.author.name if post.author else "Unknown"
        date = datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%d")
        content = post.selftext
        num_comments = post.num_comments
        doc = RedditDocument(title=title, author=author, date=date, num_comments=num_comments, content=content)
        documents.append(doc)
    
    return documents

# Récupérer les données d'Arxiv
def fetch_arxiv_data(query, max_results=100):
    url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    
    documents = []
    for entry in soup.find_all('entry'):
        title = entry.title.text
        authors = [author.text for author in entry.find_all('author')]
        date = entry.updated.text.split('T')[0]  # On prend la date mise à jour
        content = entry.summary.text
        doc = ArxivDocument(title=title, authors=authors, date=date, content=content)
        documents.append(doc)
    
    return documents

# Charger les données depuis Reddit et Arxiv
def load_corpus_from_reddit_and_arxiv():
    corpus = Corpus()

    # Charger les données de Reddit (par exemple, subreddit 'python')
    reddit_documents = fetch_reddit_data(subreddit_name='python', limit=50)  # Limité à 50 posts pour l'exemple
    for doc in reddit_documents:
        corpus.add_document(doc)

    # Charger les données d'Arxiv (par exemple, recherche 'machine learning')
    arxiv_documents = fetch_arxiv_data(query='machine+learning', max_results=50)  # Limité à 50 résultats
    for doc in arxiv_documents:
        corpus.add_document(doc)

    return corpus

# Extraire les suggestions de mots-clés à partir des documents du corpus, en excluant les stop words et symboles
def get_keyword_suggestions(corpus, top_n=10):
    # Extraire les stop words en anglais
    stop_words = set(stopwords.words('english'))
    
    # Expression régulière pour supprimer les symboles non désirés
    unwanted_symbols = re.compile(r'[^\w\s]')
    
    # Extraire tous les mots des documents du corpus en filtrant les stop words et symboles indésirables
    all_words = [
        word.lower() for doc in corpus.documents
        for word in doc.content.split()
        if word.lower() not in stop_words and not unwanted_symbols.search(word)
    ]
    
    # Créer un dictionnaire des fréquences de mots
    freq_dict = {}
    for word in all_words:
        freq_dict[word] = freq_dict.get(word, 0) + 1
    
    # Trier les mots par fréquence et retourner les top_n mots les plus fréquents
    sorted_words = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, _ in sorted_words[:top_n]]

# Récupérer les suggestions de mots-clés
corpus = load_corpus_from_reddit_and_arxiv()  # Charger le corpus avec Reddit et Arxiv
keyword_suggestions = get_keyword_suggestions(corpus, top_n=10)

# Créer un champ de saisie pour les suggestions de mots-clés
keyword_autocomplete = widgets.Combobox(
    options=keyword_suggestions, 
    description="Suggestions:", 
    placeholder="Sélectionnez un mot-clé"
)

# Afficher le widget de suggestions
display(keyword_autocomplete)

# Moteur de recherche avec affichage de la progression
class SearchEngine:
    def __init__(self, corpus):
        self.corpus = corpus

    def search(self, keyword):
        results = []
        for i, doc in tqdm(enumerate(self.corpus.documents), desc="Recherche en cours", unit="doc"):
            if keyword.lower() in doc.content.lower():
                results.append(doc)
        return results

# Initialisation du moteur de recherche
engine = SearchEngine(corpus)

# Fonction pour afficher les résultats sous forme de tableau
def display_results_as_table(results):
    table_data = []
    for doc in results:
        table_data.append([doc.title, doc.author, doc.date.strftime('%Y-%m-%d'), doc.content[:100]])  # Limité à 100 caractères
        
    df = pd.DataFrame(table_data, columns=['Titre', 'Auteur', 'Date', 'Extrait'])
    display(df)

# Interface utilisateur avec ipywidgets
# Widgets de base
title_label = widgets.Label(value="Moteur de Recherche de Documents")
keyword_text = widgets.Text(description="Mot-clé:")
num_slider = widgets.IntSlider(value=5, min=1, max=20, description="Nb Docs:")
output = widgets.Output()

# Ajout du texte avant la barre de progression
progress_label = widgets.Label(value="Barre de progression")
display(progress_label)

# Ajout de la barre de progression
progress = widgets.FloatProgress(min=0, max=100)
display(progress)

# Fonction pour effectuer la recherche avec une barre de progression
def search_button_clicked(b):
    with output:
        output.clear_output()  # Efface la sortie précédente
        keyword = keyword_text.value
        num_docs = num_slider.value
        total_docs = len(corpus.documents)
        progress.value = 0
        results = []
        
        # Recherche avec mise à jour de la barre de progression
        for i, doc in enumerate(engine.search(keyword)):
            progress.value = (i / total_docs) * 100  # Mise à jour de la progression
            if len(results) < num_docs:
                results.append(doc)
        
        # Affichage des résultats
        display_results_as_table(results[:num_docs])

# Bouton pour lancer la recherche
search_button = widgets.Button(description="Lancer la Recherche")
search_button.on_click(search_button_clicked)

# Affichage des éléments
display(title_label, keyword_text, num_slider, search_button, output)

# Widgets supplémentaires pour filtrer par date et auteur
# Récupérer tous les auteurs du corpus, en aplatissant les listes d'auteurs d'Arxiv
all_authors = [author for doc in corpus.documents for author in (doc.authors if isinstance(doc, ArxivDocument) else [doc.author])]
unique_authors = list(set(all_authors))  # Supprimer les doublons

# Créer un dropdown pour sélectionner un auteur
author_dropdown = widgets.Dropdown(options=unique_authors, description='Auteur:')
date_picker = widgets.DatePicker(description="Date")

# Fonction de recherche filtrée
def filtered_search_button_clicked(b):
    with output:
        output.clear_output()
        keyword = keyword_text.value
        author = author_dropdown.value
        date = date_picker.value
        num_docs = num_slider.value
        
        # Appliquer les filtres
        filtered_results = [
            doc for doc in engine.search(keyword)
            if (author in doc.author if author else True) and
               (date and doc.date.date() == date if date else True)
        ]
        
        display_results_as_table(filtered_results[:num_docs])  # Afficher les résultats filtrés sous forme de tableau

# bouton de recherche filtrée
filtered_search_button = widgets.Button(description="Lancer Recherche Filtrée")
filtered_search_button.on_click(filtered_search_button_clicked)

# Affichage des nouveaux widgets
display(author_dropdown, date_picker, filtered_search_button)
