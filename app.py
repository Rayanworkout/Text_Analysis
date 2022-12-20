import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import streamlit as st

from gensim.models import TfidfModel
from matplotlib.lines import Line2D
from streamlit import components

from utils import *

# TODO CORRECT DOCS + REVOIR DOWNLOAD FUNCTION + CLUSTERING

@st.cache(show_spinner=False)
def get_clusters_cached(df, k):
    return clustering(df, k)

@st.cache(show_spinner=False)
def get_top_keywords_cached(top_keywords_number, features, labels, vec):
    return get_top_keywords(top_keywords_number, features, labels, vec)


############################################################################################################

def get_parameters(choices=["full_analysis_choice"]):
    
    necessary_parameters = {
        "full_analysis_choice" : ["keywords_qty", "n", "wordcloud_number", "user_k", "top_keywords_number", "topics_number"],
        "wordcloud_choice" : ["wordcloud_number"],
        "keywords_choice" : ["keywords_qty", "n"],
        "clusters_choice": ["user_k", "top_keywords_number", "n", "keywords_qty"],
        "topics_choice": ["topics_number"]
    }
    
    parameters = {}
    
    # ASKING FOR UNWANTED WORDS
    user_unwanted_words = st.text_input("🗣 What words must be ignored in the analysis ? Put a comma between each word",
                            max_chars=1000,
                            placeholder="Apple, Louis Vuitton, Acquisition ...")
    if user_unwanted_words:
        parameters["user_unwanted_words"] = user_unwanted_words
    else:
        parameters["user_unwanted_words"] = None
    
    for element in choices:
        if "keywords_qty" in necessary_parameters[element] and "keywords_qty" not in parameters:
            # CHOOSING NUMBER OF KEYWORDS   
            keywords_qty = st.number_input("✂️ How many keywords do you want to extract ?",
                                            min_value=5,
                                            max_value=50,
                                            value=25)
            parameters["keywords_qty"] = keywords_qty
        
        if "n" in necessary_parameters[element] and "n" not in parameters:
            # CHOOSING NGRAMS SIZE
            n = st.number_input("🆖 What should be the size of N-Grams ?",
                                min_value=1,
                                max_value=4,
                                value=4)
            parameters["n"] = n
        
        if "wordcloud_number" in necessary_parameters[element] and "wordcloud_number" not in parameters:
            # CHOOSING WordCloud number of words SIZE
            wordcloud_number = st.number_input("☁️ How many words do you want to see in the WordCloud ?",
                                            min_value=50,
                                            max_value=300,
                                            value=100)
            parameters["wordcloud_number"] = wordcloud_number
            
        if "user_k" in necessary_parameters[element] and "user_k" not in parameters:
            # CHOOSING NUMBER OF CLUSTERS
            user_k = st.number_input("❓ How many Clusters do you want to extract ?",
                                    min_value=3,
                                    max_value=11)
            user_choose_k = st.checkbox("Apply custom ***k***")
            
            if user_choose_k:
                parameters["user_k"] = user_k
            else:
                parameters["user_k"] = 0
        
        if "top_keywords_number" in necessary_parameters[element] and "top_keywords_number" not in parameters:
            # CHOOSING NUMBER OF KEYWORDS FOR EACH CLUSTER
            top_keywords_number = st.number_input("📖 How many Keywords do you want to show for each cluster ?",
                                                min_value=2,
                                                max_value=10,
                                                value=6)
            parameters["top_keywords_number"] = top_keywords_number
            
        if "topics_number" in necessary_parameters[element] and "topics_number" not in parameters:
            # CHOOSING NUMBER OF TOPICS
            topics_number = st.number_input("📝 How many topics do you want to extract ?",
                                        min_value=3,
                                        max_value=30,
                                        value=15)
            parameters["topics_number"] = topics_number
        
    return parameters


def run_analysis(text):
    
    full_analysis_choice = st.checkbox("Full Analysis 🔮", value=True)
    
    wordcloud_choice = False
    keywords_choice = False
    clusters_choice = False
    topics_choice = False
    
    if not full_analysis_choice:
        wordcloud_choice = st.checkbox("Wordcloud ☁️")
        keywords_choice = st.checkbox("Keywords 🔑")
        clusters_choice = st.checkbox("Clusters 🧩")
        topics_choice = st.checkbox("Topics 📝")
    
      
    
    st.header("Parameters ⚙️")
    st.subheader("💭 You can leave default parameters if you're not sure. Check the sidebar for more informations")
    
    language = st.selectbox("🌐 **Select the language of the data**", ["🇫🇷", "🇬🇧 / 🇺🇸"])
            
    model = "fr_core_news_md" if language == "🇫🇷" else "en_core_web_sm"
    lang = "fr" if language == "🇫🇷" else "en"
    
    user_choices = []
    choices = [("full_analysis_choice", full_analysis_choice),
                    ("wordcloud_choice", wordcloud_choice),
                    ("keywords_choice", keywords_choice),
                    ("clusters_choice", clusters_choice),
                    ("topics_choice", topics_choice)]
    
    for name, value in choices:
        if value:
            user_choices.append(name)
            
            
    parameters = get_parameters(user_choices)
        
    validate_parameters = st.button("Generate Analysis", type="primary")
    
    if validate_parameters:
        
        with st.spinner("🌪 Cleaning Data ..."):
            
            nlp, stop_words = load_model_and_stopwords(model)
            
            if parameters["user_unwanted_words"]:
                unwanted_words = {word.lower().strip() for word in parameters["user_unwanted_words"].split(",")}
                stop_words.update(unwanted_words)
            
            # Cleaning each word of each sentence and putting all words in 1 big list
            all_data = ' '.join(return_tokens_by_word(nlp, stop_words, sentence) for sentence in text)
        
        if "full_analysis_choice" in user_choices or "wordcloud_choice" in user_choices:
            with st.spinner("☁️ Generating WordCloud ..."):
                
                generate_wordcloud(all_data, stop_words, parameters["wordcloud_number"])
                
                st.header("☁️ WordCloud")
                
                st.image("files/wordcloud.png")
                
        if "full_analysis_choice" in user_choices or "keywords_choice" in user_choices:
            with st.spinner("🌀 Extracting Keywords ..."):
                # USING YAKE TO EXTRACT BIGRAMS KEYWORDS
                data = extract_keywords(parameters["n"], parameters["keywords_qty"], all_data, lang)
                
                if not data:
                    st.write("# No keywords found, try again with a bigger sample ❌")
                    return
            
            with st.spinner("📊 Generating Chart ..."):
                # PLOTTING KEYWORDS
                df = pd.DataFrame(data, columns=["ngram", "score"])
                
                fig = plt.figure(figsize=(12, 35))
                plt.rcParams.update({'font.size': 25})
                plt.tick_params(axis='both', which='major', pad=15)
                
                plt.barh(df.ngram, df.score, 0.5, color='#1DA1F2')
                plt.grid(True, color='grey', linewidth=0.6, alpha=0.9)
                                
                plt.xticks([])
                
                st.header("📊 Most represented Keywords\n"
                        " **(The smaller the bar, the more relevant the keyword is)**\n\n")
                
                st.pyplot(plt)
                
        
        if "full_analysis_choice" in user_choices or "clusters_choice" in user_choices:
            with st.spinner("🌀 Extracting Keywords ..."):
                # USING YAKE TO EXTRACT BIGRAMS KEYWORDS
                data = extract_keywords(parameters["n"], parameters["keywords_qty"], all_data, lang)
                
                if not data:
                    st.write("# No keywords found, try again with a bigger sample ❌")
                    return

                df = pd.DataFrame(data, columns=["ngram", "score"])
                
            with st.spinner("🧩 Computing Clusters ..."):
                st.header("👉 Now let's create clusters from these keywords")
                
                features, labels , vec, colors, df = get_clusters_cached(df, parameters["user_k"])
                
                labels = [x + 1 for x in labels]
                
                top_keywords = get_top_keywords_cached(parameters["top_keywords_number"], features, labels, vec)
            
            with st.spinner("📊 Generating Chart ..."):
                                                                    
                # PLOTTING DATA WE GOT FROM PCA
                fig, ax = plt.subplots(1, figsize=(15,15))

                plt.scatter(df.x_coordinates, df.y_coordinates, c=df.c, alpha = 0.6, s=70)

                # PLOTTING CENTROIDS
                # plt.scatter(cen_x, cen_y, marker='^', c=colors, s=70)

                # BUILD LEGEND ELEMENT
                legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Cluster {theme}", markersize=8,
                                        markerfacecolor=colors[cluster])
                                for cluster, theme in enumerate(set(labels))]
                # ADD LEGEND ELEMENT
                plt.legend(handles=legend_elements, loc='upper right')
                
                st.subheader("How to read these informations ? 💭")
                
                st.write("For each cluster is a set of keywords that represents the main idea of the cluster.\n"
                        "By observing the chart, you can check which topics are next to each other in the document.\n"
                        "You can also see the main topics of this document by looking at the biggest clusters and their respective"
                        " topics.")
                
                plt.savefig("files/clusters.png")
                st.pyplot(plt)
                
                for i, words in enumerate(top_keywords):
                    st.subheader(f"Cluster {i + 1}")
                    st.write('- ' + ' | '.join(words))
                
        if "full_analysis_choice" in user_choices or "topics_choice" in user_choices:
            with st.spinner("Applying Topic Modeling with LDA ..."):

                rm_stop_words = clean_sentences(text, nlp, stop_words)
                
                gensim_data = [gensim.utils.simple_preprocess(text, deacc=True) for text in rm_stop_words]
                
                _, data_bigrams_trigrams, _ = bigrams_trigrams(gensim_data)
                
                # TF-IDF
                id2word = gensim.corpora.Dictionary(data_bigrams_trigrams)

                corpus = [id2word.doc2bow(text) for text in data_bigrams_trigrams if id2word.doc2bow(text)]

                tfidf = TfidfModel(corpus, id2word=id2word)

                low_value = 0.03
                words_missing_in_tfidf = []

                for i in range(0, len(corpus)):
                    bow = corpus[i]
                    low_value_words = [] #reinitialize to be safe. You can skip this.
                    tfidf_ids = [id for id, _ in tfidf[bow]]
                    bow_ids = [id for id, _ in bow]
                    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
                    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

                    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]  

                    #reassign        
                    corpus[i] = new_bow
                    
                lda_model_ = gensim.models.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=parameters["topics_number"],
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto')
                
                pyLDAvis.save_html(pyLDAvis.gensim_models.prepare(lda_model_, corpus, id2word, mds='mmds'),
                                "files/vis.html")
                
                with open('files/vis.html', 'r') as f:
                    html_string = f.read()
                
                components.v1.html(html_string, width=1300, height=800, scrolling=False)
                
                with open("files/vis.html", "rb") as file:
                    html_bytes = file.read()
                        
                st.download_button(label="▶️ *DOWNLOAD THIS VISUALIZATION (HTML)*",
                            data=html_bytes,
                            file_name="Vis.html",
                            mime='application/html')
            

st.title("Text Analysis 💬")

# FILLING SIDEBAR WITH INSTRUCTIONS AND INFORMATIONS

with open("README.md", "r") as readme:
    readme_text = readme.read()

st.sidebar.markdown(readme_text, unsafe_allow_html=True)


data_choice = st.selectbox("What do you want to analyze ?", ["📁 Local Universal Registration Document",
                                                             "🌐 PDF Document with URL",
                                                             "📝 Sample text"])

if data_choice == "📁 Local Universal Registration Document":
    available_files = [file.split(".")[0] for file in os.listdir("files/DEU") if file.endswith(".pdf")]

    if available_files:
        local_file_choice = st.selectbox("**Select a company to analyze**", available_files, index=0)
        
        with open(f"files/DEU/{local_file_choice}.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()
        
        
        st.markdown("➡️ **You can download the file to check relevant pages**")
        download_due_button = st.download_button(label="▶️ *DOWNLOAD THIS FILE*",
                            data=PDFbyte,
                            file_name=f"{local_file_choice}.pdf",
                            mime="application/pdf")
        
        start_button_local_file = st.button("***⚡️ START ⚡️***", type="primary")
        
        if start_button_local_file or "start_button_local_file" in st.session_state and st.session_state.start_button_local_file:
            
            if "start_button_local_file" not in st.session_state:
                st.session_state["start_button_local_file"] = True
            
            # ASKING FOR PAGES TO ANALYZE
            
            pages_slider = st.slider("📄 Pages to Analyze",
                                min_value=1,
                                max_value=get_pdf_pages_number(f"{local_file_choice}.pdf"),
                                value=(10, 50))
            
            text = get_text_from_pdf(f"{local_file_choice}.pdf", pages_slider[0], pages_slider[1])
            run_analysis(text)

            
    elif not available_files:
        st.write("⚠️ No file available. Please download the first")
    
    
elif data_choice == "🌐 PDF Document with URL":
    company_name = st.text_input("Name of the company / file", max_chars=40, placeholder="Apple, Louis Vuitton, History Lesson...")
                
    if company_name:
        document_url = st.text_input("URL of Universal Registration Document",
                                    placeholder="https://r.lvmh-static.com/uploads/2022/03/lvmh-deu-2021_vf.pdf")
        
        download_button_2 = st.button("Download this file")
        
        if download_button_2:
            result = download_file_with_url(company_name, document_url)
            st.write(result[0])
            time.sleep(3)

            if result[1]:
                st.experimental_rerun()
 
                
elif data_choice == "📝 Sample text":
    sample_text = st.text_area("Write your text here ⬇️", max_chars=10000)
    
    start_button_sample_text = st.button("***⚡️ START ⚡️***", type="primary")
        
    if start_button_sample_text or "start_button_sample_text" in st.session_state and st.session_state.start_button_sample_text:
        
        if "start_button_sample_text" not in st.session_state:
            st.session_state["start_button_sample_text"] = True
    
        if len(sample_text.split()) > 50:
            run_analysis(sample_text.split("."))
        
        elif start_button_sample_text and len(sample_text.split()) < 50:
            st.write("⚠️ Please write at least 50 words")
