# **WHAT IS IT ?**

***👉 This app is intended to analyze a company's Universal Registration Document 
and apply some basic Topic Modeling to it. The model extracts keywords, create clusters, generate charts and an interactive visualisation of the main topics***

# **HOW TO USE ?**

## TLDR ; select a file in the list, its language and click on "START" !

### You can install it locally by cloning the repository and running the following command in your terminal

```bash
make install

make run
``` 

-------------------------------------------

## ⚙️ Feel free to run multiple analysis with different parameters in order to get better results ⚖️
## 🧠 Note that the model is the most optimized with french language

-------------------------------------------

# 📝 Let's Start

***- The process is simple: choose a file in the list, choose its language (written next to company name) and run the analysis.***

***Before launching the analysis, you can download the file in order to select the pages you want to analyze, since some are irrelevant.***

# ⚙️ Parameters

## Unwanted Words
***You can write some words that might pollute the analysis, such as the name of the company or some highly recurring and expected words. You need to write these words or group of words with a comma between each, otherwise the program may have issues taking them into account***

## N-Grams
***N-Grams are group of words that we consider relevant. Choosing a length of 2 will execute the analysis with bigrams, 3 with trigrams and so on.***

## Clusters Number
***A cluster is a group of N-Grams that are close to each other in term of topic / meaning. You can choose the number of clusters you want to create.***

## Keywords Number
***Each cluster is represented by a set of words that helps you understand the main topic of the cluster. You can choose the number of keywords you want to extract for each cluster.***

## Topics Number
***The model will extract the main topics of the document with LDA method. You can choose the number of topics you want to extract.***

# Understanding Output

***The first elements displayed are Wordclouds. A Wordcloud is a visual representation of the most frequent words in the document. The program will generate 2, one with the wordcloud library classification and the other using the Yake library. You can then guess the main topics and the themes associated. You can refer to Yake documentation to understand how these keywords are chosen. https://github.com/LIAAD/yake***

***- After the keywords extraction, we create clusters from the previously extracted N-Grams.***

***Clusters are displayed in a scatter plot. You can observe which subjects are close to each other. Each cluster has an associated set of keywords which help you understand the main topic of the cluster.***

***Results can vary according to the parameters you chose earlier. If they aren't good enough, try running the analysis with different parameters.***

***- Then we use Latent Dirichlet Allocation (LDA) to apply topic modeling and extract the main topics of the document. The visualisation is made through pyLDAvis. https://github.com/bmabey/pyLDAvis***

***This visualisation is interactive and meant to show the topics mentionned in the document as well as the main keywords associated.***

***🧠 Techniques used***

- Machine Learning
- Natural Language Processing
- LDA
- Clustering
- N-grams
- Vector Comparison
- TF-IDF
- K-means
- Latent Dirichlet Allocation (LDA)


***💎 Libs***

- Pandas / Numpy
- Sklearn
- Gensim
- Spacy
- pyLDAvis
- Streamlit
- Matplotlib
- Fitz
- Unidecode

***Source code***

*https://github.com/Rayanworkout/text_analysis*