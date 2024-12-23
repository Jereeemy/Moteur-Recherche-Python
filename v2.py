import re
import pandas as pd
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
import json
import praw
import urllib.request
import xmltodict
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Dictionnaire pour stocker les instances de Document
id2doc = {}
id_counter = 0

reddit = praw.Reddit(
    client_id='tEQSvErg_mor-oAPoqEjlg',
    client_secret='FbQthefsNh8wiLCYlkAjeFx7L1VLeA',
    user_agent='WebScrapping'
)

# Classe de base pour les documents
class Document:
    def __init__(self, title, author, date):
        self.title = title  
        self.author = author  
        self.date = date  
        self.type = self.getType()  

    def getType(self):
        return "Document"  # Type générique pour la classe de base

    def __str__(self):
        return f"{self.title} by {self.author} on {self.date} ({self.type})"

# Classe pour les documents Reddit
class RedditDocument(Document):
    def __init__(self, title, author, date, num_comments):
        super().__init__(title, author, date)  # Appel au constructeur de la classe de base
        self.num_comments = num_comments

    def getType(self):
        return "Reddit"  

    def __str__(self):
        return super().__str__() + f", Comments: {self.num_comments}"  

# Classe pour les documents Arxiv
class ArxivDocument(Document):
    def __init__(self, title, authors, date):
        super().__init__(title, authors, date)  # Appel au constructeur de la classe de base
        self.authors = authors

    def getType(self):
        return "Arxiv"  

    def __str__(self):
        authors_str = ", ".join(self.authors)  # Conversion de la liste des auteurs en chaîne de caractères
        return super().__str__() + f", Authors: {authors_str}"

# Classe pour gérer un corpus de documents
class Corpus:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Corpus, cls).__new__(cls)
            cls._instance.documents = []  # Liste des documents dans le corpus
            cls._instance.concat_text = None  # Texte concaténé pour les recherches
        return cls._instance

    def add_document(self, document):
        self.documents.append(str(document))  # Ajout du document sous forme de chaîne dans la liste

    def display_documents(self):
        for doc in self.documents:
            print(doc)  # Affichage de tous les documents du corpus

    # Partie 1 : Expressions régulières
    def search(self, keyword):
        if self.concat_text is None:
            self.concat_text = ' '.join(self.documents)  # Concatène tous les documents pour les recherches
        return re.findall(r'\b{}\b'.format(re.escape(keyword)), self.concat_text)  # Recherche le mot-clé dans le texte concaténé

    def concorde(self, expression, context_size=30):
        if self.concat_text is None:
            self.concat_text = ' '.join(self.documents)  # Concatène tous les documents pour les recherches
        matches = re.finditer(expression, self.concat_text)  # Recherche les correspondances avec l'expression régulière
        
        concordance_data = []  # Liste pour stocker les résultats de la concordance
        for match in matches:
            start, end = match.span()  # Récupère les indices de début et de fin du match
            contexte_gauche = self.concat_text[max(0, start - context_size):start].strip()  
            contexte_droit = self.concat_text[end:end + context_size].strip() 
            concordance_data.append({
                "contexte gauche": contexte_gauche,
                "motif trouvé": match.group(),
                "contexte droit": contexte_droit
            })

        return pd.DataFrame(concordance_data)  # Retourne un DataFrame des résultats

    # Partie 2 : Statistiques textuelles
    @staticmethod
    def nettoyer_texte(texte):
        texte = texte.lower()  # Convertir tout le texte en minuscules
        texte = re.sub(r'\n', ' ', texte)  # Remplacer les nouvelles lignes par des espaces
        texte = re.sub(r'[^\w\s]', '', texte)  # Supprimer la ponctuation
        texte = re.sub(r'\d+', '', texte)  # Supprimer les chiffres
        return texte

    # 1.1 Fonction pour construire le vocabulaire
    def construire_vocabulaire(self):
        vocab = {}
        for doc in self.documents:
            mots = self.nettoyer_texte(doc).split()  # Nettoyage et séparation des mots
            for mot in mots:
                if mot not in vocab:
                    vocab[mot] = {"id": len(vocab), "occurrences": 0, "doc_count": 0}
                vocab[mot]["occurrences"] += 1  # Comptage des occurrences de chaque mot
        for mot in vocab.keys():
            vocab[mot]["doc_count"] = sum(1 for doc in self.documents if mot in doc.lower())  # Comptage des documents contenant le mot
        return vocab

    # 1.2 Construction de la matrice TF (Documents x Mots)
    def construire_matrice_TF(self):
        vocab = self.construire_vocabulaire()
        vocab_index = {mot: info["id"] for mot, info in vocab.items()}  # Indexation des mots
        nb_documents = len(self.documents)
        nb_mots_vocab = len(vocab)

        data, rows, cols = [], [], []  # Listes pour construire la matrice creuse
        for doc_idx, doc in enumerate(self.documents):
            mots = self.nettoyer_texte(doc).split()  # Nettoyage et séparation des mots
            compteur_mots = Counter(mots)  # Comptage des mots dans le document
            for mot, freq in compteur_mots.items():
                if mot in vocab_index:
                    data.append(freq)
                    rows.append(doc_idx)
                    cols.append(vocab_index[mot])

        mat_TF = csr_matrix((data, (rows, cols)), shape=(nb_documents, nb_mots_vocab), dtype=int)  # Construction de la matrice TF
        return mat_TF, vocab

    # 1.4 Construction de la matrice TFxIDF
    def construire_matrice_TFIDF(self):
        mat_TF, vocab = self.construire_matrice_TF()
        nb_documents = mat_TF.shape[0]
        idf = np.zeros(mat_TF.shape[1])
        for mot, info in vocab.items():
            index = info["id"]
            doc_count = info["doc_count"]
            idf[index] = np.log((nb_documents + 1) / (doc_count + 1)) + 1  # Calcul du IDF pour chaque mot

        mat_IDF = csr_matrix(idf)
        mat_TFIDF = mat_TF.multiply(mat_IDF) 
        return mat_TFIDF

    # 2.3 Fonction pour compter les occurrences des mots
    def compter_occurrences(self):
        vocabulaire = self.construire_vocabulaire()
        word_counts = Counter()  # Compteur pour les mots
        for doc in self.documents:
            mots = self.nettoyer_texte(doc).split()
            word_counts.update(mots)  # Mise à jour du compteur avec les mots du document
        return pd.DataFrame(word_counts.most_common(), columns=['mot', 'fréquence'])  # Retourne un DataFrame avec les mots et leur fréquence

    # 2.4 Enrichissement du tableau freq avec les fréquences de documents
    def stats(self, n=10):
        freq_df = self.compter_occurrences()
        freq_df['document frequency'] = freq_df['mot'].apply(
            lambda mot: sum(1 for doc in self.documents if mot in doc.lower())  # Calcul de la fréquence des documents
        )
        print(f"Nombre de mots différents dans le corpus : {freq_df.shape[0]}")
        print(freq_df.head(n))  # Affichage des n premiers résultats

    # 3. Moteur de recherche basé sur la similarité cosinus
    def search_cosine(self, query):
        vocab = self.construire_vocabulaire()
        vocab_index = {mot: info["id"] for mot, info in vocab.items()}  # Indexation des mots
        nb_documents = len(self.documents)

        # Convertir la requête en un vecteur TF
        query_terms = self.nettoyer_texte(query).split()
        query_vector = np.zeros(len(vocab))
        for term in query_terms:
            if term in vocab_index:
                query_vector[vocab_index[term]] = 1  # Fréquence binaire (1 ou 0)

        # Construire la matrice TF des documents
        mat_TF, _ = self.construire_matrice_TF()

        # Calculer la similarité cosinus entre la requête et chaque document
        similarities = cosine_similarity(mat_TF, query_vector.reshape(1, -1)).flatten()

        # Trier les résultats par similarité décroissante
        ranked_results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

        # Afficher les résultats
        print("\nMeilleurs résultats de la recherche :")
        for idx, score in ranked_results[:10]:
            print(f"Document {idx} avec un score de similarité : {score:.4f}")
            print(self.documents[idx])
            print()

# Fonction de collecte des données de Reddit
def collect_reddit_data():
    hot_posts = reddit.subreddit('Coronavirus')
    for idx, post in enumerate(hot_posts.hot(limit=100)):
        combined_text = f"{post.title}. {post.selftext}".replace("\n", " ")  # Texte combiné pour chaque post
        doc = RedditDocument(
            title=post.title,
            author=str(post.author) if post.author else "Inconnu",
            date=str(datetime.fromtimestamp(post.created_utc)),
            num_comments=post.num_comments
        )
        id2doc[f'reddit_{idx}'] = doc  # Ajout du document dans le dictionnaire

# Fonction de collecte des données d'Arxiv
def collect_arxiv_data():
    query = "covid"
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=100'
    time.sleep(1)  # Pause pour respecter les limites d'API
    with urllib.request.urlopen(url) as response:
        data = response.read()
    parsed_data = xmltodict.parse(data)
    for idx, entry in enumerate(parsed_data['feed']['entry']):
        auteur = entry['author'][0].get('name', "Inconnu") if 'author' in entry and isinstance(entry['author'], list) and entry['author'] else "Inconnu"
        doc = ArxivDocument(
            title=entry['title'],
            authors=[auteur],
            date=entry['published']
        )
        id2doc[f'arxiv_{idx}'] = doc  # Ajout du document dans le dictionnaire


corpus = Corpus()

# Collecte de données (Reddit et Arxiv)
collect_reddit_data()
collect_arxiv_data()

# Ajouter les documents collectés au corpus
for doc_id, doc in id2doc.items():
    corpus.add_document(doc)

# Recherche par mots-clés
mot_cle = "covid"
resultats_recherche = corpus.search(mot_cle)
print(f"Résultats de la recherche pour '{mot_cle}':")
print(resultats_recherche)

# Concordance avec le mot-clé
concordances = corpus.concorde(mot_cle, context_size=30)
print(f"Concordances pour '{mot_cle}':")
print(concordances)

# Statistiques sur le vocabulaire et la fréquence des mots
corpus.stats(n=10)

# Recherche par similarité cosinus
query = "covid"
corpus.search_cosine(query)
