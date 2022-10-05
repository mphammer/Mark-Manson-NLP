import pandas as pd
import sqlite3
from common import *
import string
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pprint import pprint
import sys

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

# Gensim
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora

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

    return " ".join(punc_free_words)

def word_cloud(long_string):
    #  Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

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

    print(df.info())

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
        
        word_tokens = simple_preprocess(cleaned_text)
        row["tokens"] = word_tokens
        article_words_list.append(word_tokens)

        long_string_list.append(cleaned_text)
    # print(len(article_words_list))
    # sys.exit(0)

    long_string = " ".join(long_string_list)
    word_cloud(long_string)

    num_articles = len(article_words_list)
    num_topics = 4
    articles_per_topic = num_articles / num_topics
    filter_words_in_less_then_n_docs = articles_per_topic // 2
    filter_words_in_more_than_n_percent_of_docs = 0.75 # (num_articles - filter_words_in_less_then_n_docs) / num_articles
    print("Num Topics: {}".format(num_topics))
    print("Total Articles: {}".format(num_articles))
    print("Articles Per Topic: {}".format(articles_per_topic))
    print("Articles Above: {}".format(filter_words_in_less_then_n_docs))
    print("Articles Below: {:.2f}".format(filter_words_in_more_than_n_percent_of_docs*num_articles))
    print("Word In more than N Articles: {:.2f}%".format(float(filter_words_in_less_then_n_docs*100/num_articles)))
    print("Word In less than N Articles: {:.2f}%".format(float(filter_words_in_more_than_n_percent_of_docs*100)))
    
        

    a = df.loc["interviewing-like-a-boss"]
    a_text = a["text"]
    # print(a_text)
    # print(article_words_list[0])

    id2word_dictionary = corpora.Dictionary(article_words_list)
    #less than no_below documents (absolute number) or
    #more than no_above documents (fraction of total corpus size, not absolute number).
    #after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
    id2word_dictionary.filter_extremes(no_below=filter_words_in_less_then_n_docs, no_above=filter_words_in_more_than_n_percent_of_docs, keep_n=5000)

    bag_of_words_corpus = [id2word_dictionary.doc2bow(article_words) for article_words in article_words_list]
    # print(bag_of_words_corpus)
    

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=bag_of_words_corpus,
                                        id2word=id2word_dictionary,
                                        num_topics=num_topics,
                                        passes=50)

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics(num_topics=num_topics, num_words=8))
    # doc_lda = lda_model[corpus]

    