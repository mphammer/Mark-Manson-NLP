import pandas as pd
import sqlite3
from common import *
import string
import re
import sys
import mywordcloud
import LatentDirichletAllocation

# $ pip install nltk
# $ python
# >> nltk.download('stopwords')
# >> nltk.download('punkt')
# >> nltk.download('wordnet')
# >> nltk.download('omw-1.4')
# quit()
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from gensim.utils import simple_preprocess



def get_file_text(filepath):
    '''
    This function joins all the lines into a single string and removes any newlines
    '''
    lines = []
    with open(filepath) as f:
        lines = f.readlines()
    return " ".join(lines).replace("\n", " ")

def clean_text(text):
    # lowercase the words
    lowercase_text = text.lower()
    
    # convert text into a list of words (tokens)
    word_tokens = word_tokenize(lowercase_text)
    
    # Remove Stop Words
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    stop_free_words = [w for w in word_tokens if w not in stop_words]

    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lammatized_words = [lemmatizer.lemmatize(w) for w in stop_free_words]

    # Stem the words
    # ps = PorterStemmer()
    # stemmed_words = [ps.stem(w) for w in lammatized_words]

    # remove the punctuation
    punc_list = set(string.punctuation)
    punc_free_words = [w for w in lammatized_words if w not in punc_list]
    alphanumeric_words = [w for w in punc_free_words if (w.isalnum())]

    return " ".join(alphanumeric_words)
    

def word_count(words_list):
    word_counts = {}
    for word in words_list:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_count

if __name__ == "__main__":
    # Read sqlite query results into a pandas DataFrame
    con = sqlite3.connect(DATABASE_NAME)
    df = pd.read_sql_query("SELECT * from articles", con, index_col ="name")
    con.close()

    # Create Text Column
    df["text"] = ""
    df["tokens"] = ""
    long_string_list = []
    article_words_list = []
    for i, row in df.iterrows():
        filepath = row["filepath"]
        text = get_file_text(filepath)
        cleaned_text = clean_text(text)
        row["text"] = cleaned_text
        
        word_tokens = word_tokenize(cleaned_text)
        row["tokens"] = word_tokens
        article_words_list.append(word_tokens)

        long_string_list.append(cleaned_text)

    # long_string = " ".join(long_string_list)
    # mywordcloud.create_word_cloud(long_string)

    # a = df.loc["interviewing-like-a-boss"]
    # a_text = a["text"]
    # print(a_text)
    # print(article_words_list[0])
    # sys.exit(0)

    num_articles = len(article_words_list)
    num_topics = 4
    articles_per_topic = num_articles / num_topics
    filter_words_in_less_then_n_docs = articles_per_topic // 2
    filter_words_in_more_than_n_percent_of_docs = 0.75
    print("Num Topics: {}".format(num_topics))
    print("Total Articles: {}".format(num_articles))
    print("Articles Per Topic: {}".format(articles_per_topic))
    print("Articles Above: {}".format(filter_words_in_less_then_n_docs))
    print("Articles Below: {:.2f}".format(filter_words_in_more_than_n_percent_of_docs*num_articles))
    print("Word In more than N Articles: {:.2f}%".format(float(filter_words_in_less_then_n_docs*100/num_articles)))
    print("Word In less than N Articles: {:.2f}%".format(float(filter_words_in_more_than_n_percent_of_docs*100)))
    print()
    
    # LatentDirichletAllocation.latent_dirishlet_allocation(article_words_list, num_topics, filter_words_in_less_then_n_docs, filter_words_in_more_than_n_percent_of_docs)

    # LatentDirichletAllocation.optimize_lda(article_words_list, [5,6], [30, 20, 10, 5], [0.55, 0.60, 0.65])
    LatentDirichletAllocation.optimize_lda(article_words_list, [6], [5], [0.55])
    