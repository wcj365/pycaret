#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_selection

#import ssl
#try:
#     _create_unverified_https_context =     ssl._create_unverified_context
#except AttributeError:
#     pass
#else:
#    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download("wordnet")
nltk.download('omw-1.4')


'''
Normalize a Text.
:parameter
    :param text: string - the text
    :param stemm: str - stemming method. None for not no stemming.
    :param lemm: str - lemmitisation method. None for no lemmatization.
    :param stopwords: list - nltk for using NLTK stop words. 
:return
    normalized text
'''
def normalize(text, stemm="porter", lemm="wordnet", stopwords="nltk"):

    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    list_words = text.split()

    ## remove Stopwords
    if stopwords == "nltk":
        list_stopwords = nltk.corpus.stopwords.words("english")
        list_words = [word for word in list_words if word not in list_stopwords]
    elif stopwords == "spacy":
        pass
    elif stopwords == "gensim":
        pass
    else:
        pass
                
    ## Stemming (remove -ing, -ly, ...)
    if stemm == "porter":
        ps = nltk.stem.porter.PorterStemmer()
        list_words = [ps.stem(word) for word in list_words]
    elif stemm == "snowball":
        ps = nltk.stem.snowball.SnowballStemmer("english")
        list_words = [ps.stem(word) for word in list_words]
    else:
        pass
                
    ## Lemmatisation (convert the word into root word)
    if lemm == "wordnet":
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        list_words = [lem.lemmatize(word) for word in list_words]
    else:
        pass

    return " ".join(list_words)


# sklearn vectorizor (Count or TF-IDF) also has a parameter "max_features".
# Can be tuned as well.

def vectorize(df, text_column, vectorizer, ngram_min, ngram_max, min_df, max_df):

    n_range = (ngram_min, ngram_max)
    
    
    if vectorizer == "count":
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=n_range)
    elif vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=n_range)

    sparse_matrix = vectorizer.fit_transform(df[text_column])
    
    X_names = vectorizer.get_feature_names()

    return pd.DataFrame(sparse_matrix.toarray(), columns=X_names, index=df.index)


def chi_squared(df, label_column, p_value_min):

    y = df[label_column]
    df_X = df.drop(columns=[label_column])
    X_names = list(df_X.columns)
    sparse_matrix = csr_matrix(df_X)
    df_features = pd.DataFrame()
    
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(sparse_matrix, y==cat)
        df_features = df_features.append(pd.DataFrame(
                       {"feature":X_names, "score":1-p, "y":cat}))
        df_features = df_features.sort_values(["y","score"], 
                        ascending=[True,False])
        df_features = df_features[df_features["score"]>p_value_min]
        
    X_names = df_features["feature"].unique().tolist()

    print(len(X_names))
    print(X_names)
    
    X_names.append(label_column)
 
    return df[X_names]
    

