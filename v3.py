# si nécéssaire
# !pip install praw
import praw
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Classe pour les documents (similaire à la V2)
class Document:
    def __init__(self, title, author, date, content=None):
        self.title = title
        self.author = author
        self.date = datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date
        self.content = content if content else ""

    def __repr__(self):
        return f"Document(title='{self.title}', author='{self.author}', date='{self.date}', content='{self.content[:100]}...')"

# Classe pour les documents Reddit (similaire à la V2)
class RedditDocument(Document):
    def __init__(self, title, author, date, num_comments, content=None):
        super().__init__(title, author, date, content)
        self.num_comments = num_comments

    def getType(self):
        return "Reddit"

    def __repr__(self):
        return f"RedditDocument(title='{self.title}', author='{self.author}', date='{self.date}', num_comments={self.num_comments}, content='{self.content[:100]}...')"

# Classe pour les documents Arxiv (similaire à la V2)
class ArxivDocument(Document):
    def __init__(self, title, authors, date, content=None):
        super().__init__(title, authors, date, content)
        self.authors = authors

    def getType(self):
        return "Arxiv"

    def __repr__(self):
        authors_str = ", ".join(self.authors)
        return f"ArxivDocument(title='{self.title}', authors='{authors_str}', date='{self.date}', content='{self.content[:100]}...')"

# Classe pour gérer un corpus de documents (similaire à la V2)
class Corpus:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Corpus, cls).__new__(cls)
            cls._instance.documents = []
            cls._instance.concat_text = None
        return cls._instance

    def add_document(self, document):
        self.documents.append(document)

    def display_documents(self):
        for doc in self.documents:
            print(doc)

    def search(self, keyword):
        if self.concat_text is None:
            self.concat_text = ' '.join([str(doc) for doc in self.documents])
        return re.findall(r'\b{}\b'.format(re.escape(keyword)), self.concat_text)

    def concorde(self, expression, context_size=30):
        if self.concat_text is None:
            self.concat_text = ' '.join([str(doc) for doc in self.documents])
        matches = re.finditer(expression, self.concat_text)
        concordance_data = []
        for match in matches:
            start, end = match.span()
            contexte_gauche = self.concat_text[max(0, start - context_size):start].strip()
            contexte_droit = self.concat_text[end:end + context_size].strip()
            concordance_data.append({
                "contexte gauche": contexte_gauche,
                "motif trouvé": match.group(),
                "contexte droit": contexte_droit
            })
        return pd.DataFrame(concordance_data)

    @staticmethod
    def nettoyer_texte(texte):
        texte = texte.lower()
        texte = re.sub(r'\n', ' ', texte)
        texte = re.sub(r'[^\w\s]', '', texte)
        texte = re.sub(r'\d+', '', texte)
        return texte

    def construire_vocabulaire(self):
        vocab = {}
        for doc in self.documents:
            mots = self.nettoyer_texte(str(doc)).split()
            for mot in mots:
                if mot not in vocab:
                    vocab[mot] = {"id": len(vocab), "occurrences": 0, "doc_count": 0}
                vocab[mot]["occurrences"] += 1
        for mot in vocab.keys():
            vocab[mot]["doc_count"] = sum(1 for doc in self.documents if mot in str(doc).lower())
        return vocab

    def construire_matrice_TF(self):
        vocab = self.construire_vocabulaire()
        vocab_index = {mot: info["id"] for mot, info in vocab.items()}
        nb_documents = len(self.documents)
        nb_mots_vocab = len(vocab)
        data, rows, cols = [], [], []
        for doc_idx, doc in enumerate(self.documents):
            mots = self.nettoyer_texte(str(doc)).split()
            compteur_mots = Counter(mots)
            for mot, freq in compteur_mots.items():
                if mot in vocab_index:
                    data.append(freq)
                    rows.append(doc_idx)
                    cols.append(vocab_index[mot])
        mat_TF = csr_matrix((data, (rows, cols)), shape=(nb_documents, nb_mots_vocab), dtype=int)
        return mat_TF, vocab

    def construire_matrice_TFIDF(self):
        mat_TF, vocab = self.construire_matrice_TF()
        nb_documents = mat_TF.shape[0]
        idf = np.zeros(mat_TF.shape[1])
        for mot, info in vocab.items():
            index = info["id"]
            doc_count = info["doc_count"]
            idf[index] = np.log((nb_documents + 1) / (doc_count + 1)) + 1
        mat_IDF = csr_matrix(idf)
        mat_TFIDF = mat_TF.multiply(mat_IDF)
        return mat_TFIDF

    def compter_occurrences(self):
        vocabulaire = self.construire_vocabulaire()
        word_counts = Counter()
        for doc in self.documents:
            mots = self.nettoyer_texte(str(doc)).split()
            word_counts.update(mots)
        return pd.DataFrame(word_counts.most_common(), columns=['mot', 'fréquence'])

    def stats(self, n=10):
        freq_df = self.compter_occurrences()
        freq_df['document frequency'] = freq_df['mot'].apply(
            lambda mot: sum(1 for doc in self.documents if mot in str(doc).lower())
        )
        print(f"Nombre de mots différents dans le corpus : {freq_df.shape[0]}")
        print(freq_df.head(n))

    def search_cosine(self, query):
        vocab = self.construire_vocabulaire()
        vocab_index = {mot: info["id"] for mot, info in vocab.items()}
        nb_documents = len(self.documents)
        query_terms = self.nettoyer_texte(query).split()
        query_vector = np.zeros(len(vocab))
        for term in query_terms:
            if term in vocab_index:
                query_vector[vocab_index[term]] = 1
        mat_TF, _ = self.construire_matrice_TF()
        similarities = cosine_similarity(mat_TF, query_vector.reshape(1, -1)).flatten()
        ranked_results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        print("\nMeilleurs résultats de la recherche :")
        for idx, score in ranked_results[:10]:
            print(f"Document {idx} avec un score de similarité : {score:.4f}")
            print(self.documents[idx])
            print()

# similaire à collect_reddit_data() 
def fetch_reddit_data(subreddit_name, limit=100):
    reddit = praw.Reddit(client_id='tEQSvErg_mor-oAPoqEjlg', client_secret='FbQthefsNh8wiLCYlkAjeFx7L1VLeA', user_agent='WebScrapping')
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

# similaire à collect_arxiv_data()
def fetch_arxiv_data(query, max_results=100):
    url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    documents = []
    for entry in soup.find_all('entry'):
        title = entry.title.text
        authors = [author.text for author in entry.find_all('author')] 
        date = entry.updated.text.split('T')[0]
        content = entry.summary.text
        doc = ArxivDocument(title=title, authors=authors, date=date, content=content)
        documents.append(doc)
    return documents

# Récupère des documents depuis Reddit et Arxiv, puis les ajoute à un corpus.
def load_corpus_from_reddit_and_arxiv():
    # Initialisation du corpus
    corpus = Corpus()
    # Récupération des documents Reddit et ajout au corpus
    reddit_documents = fetch_reddit_data(subreddit_name='python', limit=50)
    for doc in reddit_documents:
        corpus.add_document(doc)
    # Récupération des documents Arxiv et ajout au corpus
    arxiv_documents = fetch_arxiv_data(query='machine+learning', max_results=50)
    for doc in arxiv_documents:
        corpus.add_document(doc)
    return corpus

# Extrait les mots-clés les plus fréquents d'un corpus
def get_keyword_suggestions(corpus, top_n=10):
    stop_words = set(stopwords.words('english')) 
    unwanted_symbols = re.compile(r'[^\w\s]')  # Expression régulière pour filtrer les symboles indésirables
    # Création d'une liste de tous les mots dans le corpus
    all_words = [
        word.lower()  # Conversion des mots en minuscules pour uniformité
        for doc in corpus.documents
        for word in str(doc).split() # Fractionnement du texte du document en mots
        if word.lower() not in stop_words and not unwanted_symbols.search(word)
    ]
    # Création d'un dictionnaire pour stocker la fréquence de chaque mot
    freq_dict = {}
    for word in all_words:
        freq_dict[word] = freq_dict.get(word, 0) + 1  # Comptage de la fréquence des mots
    sorted_words = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)  # Tri des mots par fréquence
    return [word for word, _ in sorted_words[:top_n]]  # Retour des mots les plus fréquents

class SearchEngine:
    def __init__(self, corpus):
        self.corpus = corpus

    # Méthode de recherche dans le corpus basée sur un mot-clé
    def search(self, keyword, progress_bar):
        results = [] # Liste pour stocker les documents correspondants à la recherche
        total_docs = len(self.corpus.documents)
        for i, doc in enumerate(self.corpus.documents):
            if keyword.lower() in str(doc).lower():  # Recherche du mot-clé dans chaque document
                results.append(doc)
            # Mise à jour de la barre de progression
            progress = (i + 1) / total_docs * 100
            progress_bar.value = progress
        return results

corpus = load_corpus_from_reddit_and_arxiv()

keyword_suggestions = get_keyword_suggestions(corpus, top_n=10) # Récupération des suggestions de mots-clés

# Création du widget d'auto-complétion pour les mots-clés
keyword_autocomplete = widgets.Combobox(
    options=keyword_suggestions,
    description="Suggestions:",
    placeholder="Sélectionnez un mot-clé"
)
display(keyword_autocomplete)

engine = SearchEngine(corpus)

# Affiche les résultats sous forme de tableau en affichant les informations principales des documents Reddit et Arxiv.
def display_results_as_table(results):
    table_data = []
    for doc in results:
        if isinstance(doc, RedditDocument):
            # Ajout des informations de RedditDocument au tableau
            table_data.append([doc.title, doc.author, doc.date.strftime('%Y-%m-%d'), doc.content[:100]])
        elif isinstance(doc, ArxivDocument):
            # Joindre les auteurs d'Arxiv et ajouter les informations au tableau
            authors = ", ".join(doc.authors).strip()
            table_data.append([doc.title, authors, doc.date.strftime('%Y-%m-%d'), doc.content[:100]])
    # Création d'un DataFrame Pandas pour afficher les résultats sous forme de tableau
    df = pd.DataFrame(table_data, columns=['Titre', 'Auteur', 'Date', 'Extrait'])
    display(df)

# Création des widgets pour l'interface utilisateur
title_label = widgets.Label(value="Moteur de Recherche de Documents")
keyword_text = widgets.Text(description="Mot-clé:")
num_slider = widgets.IntSlider(value=5, min=1, max=100, description="Nb Docs:")
output = widgets.Output()
progress_label = widgets.Label(value="Barre de progression:")
progress_bar = widgets.IntProgress(value=0, min=0, max=100, description="Progression:")
search_button = widgets.Button(description="Rechercher")

# Fonction appelée lorsque l'utilisateur clique sur le bouton de recherche
def on_search_button_clicked(b):
    keyword = keyword_text.value # Récupération de la valeur saisie par l'utilisateur
    num_results = num_slider.value  # Récupération de la valeur du curseur
    progress_bar.value = 0
    with output: # Utilisation de l'objet 'output' pour afficher les résultats dans l'interface
        clear_output(wait=True)
        print(f"Recherche pour '{keyword}' (max {num_results} résultats):")
        # Lancement de la recherche des documents contenant le mot-clé, tout en mettant à jour la barre de progression
        results = engine.search(keyword, progress_bar)[:num_results]
        display_results_as_table(results)

# Lier la fonction 'on_search_button_clicked' au clic du bouton de recherche
search_button.on_click(on_search_button_clicked)

# Récupération de tous les auteurs uniques dans le corpus
all_authors = [author for doc in corpus.documents for author in (doc.authors if isinstance(doc, ArxivDocument) else [doc.author])]
unique_authors = list(set(all_authors))

author_dropdown = widgets.Dropdown(options=unique_authors, description='Auteur:')
date_picker = widgets.DatePicker(description="Date")

# Fonction appelée lorsque l'utilisateur clique sur le bouton de recherche filtrée
def filtered_search_button_clicked(b):
    keyword = keyword_text.value
    author = author_dropdown.value
    date = date_picker.value
    num_docs = num_slider.value
    progress_bar.value = 0
    
    with output:
        clear_output(wait=True)
        print(f"Recherche filtrée pour '{keyword}' (max {num_docs} résultats):")
        all_results = engine.search(keyword, progress_bar)
        # Filtrage des résultats selon l'auteur sélectionné et la date choisie
        filtered_results = [
            doc for doc in all_results
            # Vérification que l'auteur du document correspond à l'auteur sélectionné et que la date du document correspond à la date sélectionnée
            if (author in (", ".join(doc.authors) if isinstance(doc, ArxivDocument) else doc.author)) and
               (date and doc.date.date() == date if date else True)
        ]
        display_results_as_table(filtered_results[:num_docs])

filtered_search_button = widgets.Button(description="Recherche Filtrée")
filtered_search_button.on_click(filtered_search_button_clicked)

# Bouton pour afficher la concordance
concordance_button = widgets.Button(description="Afficher Concordance")
concordance_output = widgets.Output()

# Gère l'événement de clic sur le bouton de concordance pour afficher les résultats
def on_concordance_button_clicked(b):
    keyword = keyword_text.value
    with concordance_output:
        clear_output(wait=True)
        # Affichage de la concordance du mot-clé dans le corpus
        concordance_results = corpus.concorde(keyword)
        display(concordance_results)

concordance_button.on_click(on_concordance_button_clicked)

# Bouton pour afficher les statistiques du corpus
stats_button = widgets.Button(description="Afficher Statistiques")
stats_output = widgets.Output()

# Gère l'événement de clic sur le bouton des statistiques pour afficher les statistiques du corpus
def on_stats_button_clicked(b):
    with stats_output:
        clear_output(wait=True)
        # Affichage des statistiques du corpus
        corpus.stats()

stats_button.on_click(on_stats_button_clicked)

# Recherche par similarité cosinus
cosine_search_button = widgets.Button(description="Recherche Cosinus")
cosine_search_output = widgets.Output()

# Gère l'événement de clic sur le bouton de recherche par similarité cosinus
def on_cosine_search_button_clicked(b):
    query = keyword_text.value
    with cosine_search_output:
        clear_output(wait=True)
        # Recherche par similarité cosinus du mot-clé
        corpus.search_cosine(query)

cosine_search_button.on_click(on_cosine_search_button_clicked)

# Affichage des widgets dans l'interface utilisateur
display(title_label, keyword_text, num_slider, search_button, author_dropdown, date_picker, filtered_search_button, progress_label, progress_bar, output, concordance_button, concordance_output, stats_button, stats_output, cosine_search_button, cosine_search_output)
