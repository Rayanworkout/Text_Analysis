import os
import random
import re

import fitz
import gensim
import gensim.corpora
import numpy as np
import pandas as pd
import requests
import spacy
import yake

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from unidecode import unidecode
from wordcloud import WordCloud
    

def download_file_with_url(company_name, url):
    if not re.match(r"^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$", url):
        return "⚠️ This is not a valid URL, try again.", False
    try:
        with open(f"files/DEU/{company_name}.pdf", "wb") as file:
            file.write(requests.get(url).content)
        
        with requests.get(url, stream=True) as r, open(f"files/DEU/{company_name}.pdf", 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                # writing one chunk at a time to a file
                if chunk:
                    f.write(chunk)
    except Exception:
        return "Could not download this page. The URL might be invalid.", False
    
    return f'Download Successful ✅', True


def get_text_from_pdf(filename, start_page, end_page):
    pdf_file = fitz.open(f"files/DEU/{filename}")
    
    # GET A LOWERCASE TEXT VARIABLE WITH PAGES CONTENT
    text = ""
    for x in range(start_page, end_page):
        page = pdf_file.load_page(x).get_text()
        text += page.lower()

    return text.lower().split('.')


def get_pdf_pages_number(filename):
    pdf_file = fitz.open(f"files/DEU/{filename}")
    for count, _ in enumerate(pdf_file):
        pass
    
    return count


def load_model_and_stopwords(model='fr_core_news_md'):
    
    nlp = spacy.load(model)
    
    stop_words = nlp.Defaults.stop_words
        
    return nlp, stop_words  


def clean_sentences(data: list, nlp, stop_words) -> list:
    """Function to remove stop words and unwanted words from
    a list of sentences. It returns list with all cleaned sentences"""
    
    clean_data = list()
    for sentence in data:
        clean_sentence = list()
        for word in nlp(sentence):
            clean_word = unidecode(word.lemma_).lower().strip()
            if clean_word not in stop_words \
                and not word.is_punct \
                    and not clean_word.isspace() \
                        and not clean_word.isdigit() \
                            and clean_word:
                
                clean_sentence.append(clean_word)
        clean_data.append(' '.join(clean_sentence))
        
    return clean_data


def return_tokens_by_word(nlp, stop_words, sentence: str):
    return ' '.join((unidecode(word.lemma_) for word in nlp(sentence) 
            if not word.is_punct
            and len(word.lemma_) > 2
            and not word.lemma_.isspace()
            and not word.lemma_.isdigit()
            and word.lemma_ not in stop_words))


def extract_keywords(n, top, all_data, lang='fr'):
    kw_extractor = yake.KeywordExtractor(lan=lang, n=n, dedupLim=0.9, top=top)

    keywords = kw_extractor.extract_keywords(all_data)
    
    # CREATING A LIST WITH KEYWORDS AND THEIR SCORE (BASED ON OCCURENCE)
    # THE LOWER THE SCORE, THE MORE RELEVANT THE NGRAM IS.
    return sorted([(kw, score) for kw, score in keywords], key=lambda x: x[1], reverse=True)


def get_silhouette_score(X, k):
    """This function returns the best number of clusters based on the silhouette score."""
    scores = list()
    for n_clusters in range(2, k):
        clusterer = KMeans(init="k-means++", n_clusters=n_clusters, random_state=42)
        y = clusterer.fit_predict(X)
    
        scores.append((n_clusters, silhouette_score(X, y)))
    return max(scores, key=lambda x: x[1])[0]  


def clustering(df, user_k=0):
    # VECTORIZE DATA WITH TF-IDF IN ORDER TO CREATE CLUSTERS WITH THESE BIGRAMS / TRIGRAMS
    vec = TfidfVectorizer(
        ngram_range=(1, 4)
    )
    features = vec.fit_transform(df.ngram)
    
    # INITIALIZE KMeans and ideal_k
    k = get_silhouette_score(features, 10) if user_k == 0 else user_k

    km = KMeans(n_clusters=k)

    # CREATING CLUSTERS AND UPGRADING DF
    df["cluster"] = km.fit_predict(features)

    # DEFINING CENTROIDS (X AND Y COORDINATES)
    centroids = km.cluster_centers_
    cen_x = centroids[:, 0]
    cen_y = centroids[:, 1]

    # ADDING EACH CLUSTER ITS CENTROIDS COORDINATES AND COLOR (1 CLUSTER = 1 COLOR)
    df['cen_x'] = df.cluster.map({count: value for count, value in enumerate(cen_x)})
    df['cen_y'] = df.cluster.map({count: value for count, value in enumerate(cen_y)})

    colors = ['#DF2020', '#81DF20', '#2095DF', '#ff0000', '#ffa500',
            '#ffff00', '#008000', '#0000ff', '#ee82ee', '#4b0082', '#109fac']

    # MAKE COLOR LIST THE SAME SIZE AS CLUSTERS NUMBER
    colors = colors[:k]

    df['c'] = df.cluster.map({count: color for count, color in enumerate(colors)})
    
    # PCA IS USED TO MINIMIZE THE QTY OF DATA TO PLOT, HERE USING 2 COMPONENTS
    pca = PCA(n_components=2, random_state=42)

    # WE GIVE OUR VECTORIZED FEATURES TO PCA AND STORE A REDUCED VERSION IN ANOTHER VARIABLE
    pca_vecs = pca.fit_transform(features.toarray())

    # WE GET OUR COORDINATES FROM THE 2D ARRAY
    x_coordinates = pca_vecs[:, 0]
    y_coordinates = pca_vecs[:, 1]

    df['x_coordinates'] = x_coordinates
    df['y_coordinates'] = y_coordinates

    return features, km.labels_, vec, colors, df


def get_top_keywords(n_keywords, features, clusters, vec):
    """This function returns the keywords for each centroid of the KMeans
    n_keywords = the number of keywords you want.
    clusters = km.labels_, a list with all clusters
    vec = TfidfVectorizer object
    
    The purpose is to identify clusters names and apply them to chart legend."""
    
    result = []
    
    # groups the TF-IDF vector by cluster
    df = pd.DataFrame(features.todense()).groupby(clusters).mean()
    terms = vec.get_feature_names_out() # access tf-idf terms
    
    for _, value in df.iterrows():
        result.append({f'*{terms[t].upper()}*' for t in np.argsort(value)[-n_keywords:]})
        
    return result[:len(set(clusters))]


def bigrams_trigrams(gensim_data):
    # BIGRAMS & TRIGRAMS
    bigrams_phrases = gensim.models.Phrases(gensim_data, min_count=3, threshold=30)
    trigrams_phrases = gensim.models.Phrases(bigrams_phrases[gensim_data], threshold=30)

    bigram = gensim.models.phrases.Phraser(bigrams_phrases)
    trigram = gensim.models.phrases.Phraser(trigrams_phrases)

    def make_bigrams(texts):
        return list(bigram[doc] for doc in texts)

    def make_trigrams(texts):
        return list(trigram[bigram[doc]] for doc in texts)

    data_bigrams = make_bigrams(gensim_data)
    data_bigrams_trigrams = make_trigrams(data_bigrams)

    bigrams_and_trigrams = {word for entry in data_bigrams_trigrams for word in entry if "_" in word}
    
    return data_bigrams, data_bigrams_trigrams, bigrams_and_trigrams


def generate_wordcloud(text, stop_words, max_words=200):
    
    file = f'files/masks/{random.choice(os.listdir("files/masks"))}'
    mask = np.array(Image.open(file))
    
    wc = WordCloud(
        background_color='black',
        height=1000,
        width=1000,
        max_words=max_words,
        stopwords=stop_words,
        mask=mask,
        colormap='rainbow'
    )
    
    wc.generate(text.lower()).to_file(f"files/wordcloud.png")
