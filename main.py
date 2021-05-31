import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from pathlib import Path


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.kl import KLSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import streamlit as st
import pickle
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def remove_stopwords(sen: str) -> str:
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def textRank(text: str, language="english"):
    if not Path('word-embebed-en.bin').is_file():
        print(f"baixe o arquivo de vetores de palavras em https://drive.google.com/file/d/14ldEW28U7vedEwFTtUqPZGmtUomv1XmL/view e coloque na raiz do projeto")
        exit(1)
    with open('word-embebed-en.bin', 'rb') as file:
        dict_embebed = pickle.load(file)
    from nltk.tokenize import sent_tokenize

    input_text = sent_tokenize(text.lower())
    clean_sentences = pd.Series(input_text).str.replace("[^a-zãáàèéçíìõòóôêúù]", " ")
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([dict_embebed.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            
            v = np.zeros((100,))
        vectors.append(v)

    matrix_similaridade = np.zeros([len(clean_sentences), len(clean_sentences)])
    for i in range(len(input_text)):
        for j in range(len(input_text)):
            if i != j:
                matrix_similaridade[i][j] = cosine_similarity(vectors[i].reshape(1,100), vectors[j].reshape(1,100))[0,0]
    
    nx_graph = nx.from_numpy_array(matrix_similaridade)
    scores = nx.Textrank(nx_graph)
    rank = sorted(((scores[i],s) for i,s in enumerate(input_text)), reverse=True)
    result = ""
    for i in range(5):
        result += rank[i][1]
    return result
    


# https://github.com/yongzhuo/nlg-yongzhuo/blob/master/nlg_yongzhuo/text_summarization/extractive_sum/topic_base/topic_nmf.py


def nmf(text, language="english", hyperparameters={}):
    n_components = 10
    n_features = 25
    stop_words = stopwords.words(language)
    sentences = sent_tokenize(text)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7,
                                       min_df=5,
                                       max_features=n_features,
                                       stop_words=stop_words)

    tfidf = tfidf_vectorizer.fit_transform(sentences)
    model = NMF(n_components=n_components, max_iter=320)
    __package__ = model.fit_transform(tfidf.T)
    H = model.components_

    relevant_sentences = np.argsort(H.sum(axis=0))
    selected_sentences = []

    for i in range(n_components):
        index = relevant_sentences[i]
        selected_sentences.append(sentences[index])
    return "\n".join(list(selected_sentences))


def kl(text, language, sentences_count=10, **kwargs):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    return "\n".join([str(sentence) for sentence in summarizer(parser.document, sentences_count)])


def dummy(text, **kwargs):
    return text


models = {
    "NMF": nmf,
    "TextRank": textRank,
    "KL": kl}

descriptions = {
    "NMF": "The chosen model is NMF whitch differs from LSA by producing positive values for each topic, increasing interpretability in the results.",
    "TextRank": "Google Page ranks algorithm",
    "KL": "Kullback–Leibler Model implemented using sumy"
}

option = st.sidebar.selectbox(
    'Choose a model to summarize the corpus', tuple(models.keys()))

language = st.sidebar.selectbox(
    'Choose a language to summarize', ("English", "Portuguese"))

readme = f"""
# Summarizer
{descriptions[option]}
"""
st.markdown(readme)

input_text = st.text_input("Enter the text to be summarized")
button = st.button('Summarize!')

if button:
    if not input_text:
        st.error("You must insert a text before continuing")
    else:
        with st.spinner('Wait for it...'):
            result = models[option](input_text, language=language.lower())

        st.write(result)
