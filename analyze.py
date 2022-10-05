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

    # LatentDirichletAllocation.optimize_lda(article_words_list, [5,6], [30, 20, 10, 5], [0.55, 0.60, 0.65])
    # LatentDirichletAllocation.optimize_lda(article_words_list, [6], [5], [0.55])

    num_topics = 6
    no_below = 25
    no_above = 0.50
    lda_model, coherence, _ = LatentDirichletAllocation.latent_dirishlet_allocation(article_words_list, num_topics, no_below, no_above)
    print("Coherence: {}".format(coherence))
    topics_map = {}
    topics = lda_model.show_topics(num_topics=num_topics, num_words=12, formatted=False)
    for topic in topics:
        topic_id = topic[0]
        words_list = topic[1]
        words_string = " ".join([w[0] for w in words_list])
        print("{}: {}".format(topic_id, words_string))
        topics_map[topic_id] = {
            "words": topic[1],
            "clean_words": words_string
        }
    
    import gensim
    import gensim.corpora as corpora
    id2word_dictionary = corpora.Dictionary(article_words_list)
    id2word_dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=5000)

    for i, row in df.iterrows():
        article_word_tokens = row["tokens"]
        article_bag_of_words = id2word_dictionary.doc2bow(article_word_tokens)
        article_topic_distribution = lda_model.get_document_topics(article_bag_of_words, per_word_topics=False)
        max_tuple = max(article_topic_distribution, key=lambda x:x[1])
        max_topic_id = max_tuple[0]
        # print(max_topic_id)
        filepath = row["filepath"]
        if "articles" not in topics_map[max_topic_id]:
            topics_map[max_topic_id]["articles"] = []
        topics_map[max_topic_id]["articles"].append(filepath)

    for key, value in topics_map.items():
        print("Topic: {} {}".format(key, value["clean_words"]))
        articles = value["articles"]
        for a in articles:
            print(" - {}".format(a))
    
    
    
    
    # import gensim
    # from gensim.test.utils import datapath
    # temp_file = datapath("lda_model")
    # # lda_model.save(temp_file)
    # lda_model = gensim.models.LdaMulticore.load(temp_file)

    # topics = lda_model.show_topics(num_topics=6, num_words=16, formatted=False)
    # topics_map = {}
    # for topic in topics:
    #     topic_id = topic[0]
    #     words_list = topic[1]
    #     words_string = " ".join([w[0] for w in words_list])
    #     print("{}: {}".format(topic_id, words_string))
    #     topics_map[topic_id] = {
    #         "words": topic[1]
    #     }
    # topics_map[0]["Name"] = "Relationship and Dating Advice"
    # topics_map[1]["Name"] = "News"
    # topics_map[2]["Name"] = "Happiness and Philosophy"
    # topics_map[3]["Name"] = "Sexual Culture"
    # topics_map[4]["Name"] = "Emotions"
    # topics_map[5]["Name"] = "Values"
    

    # import gensim
    # import gensim.corpora as corpora
    # id2word_dictionary = corpora.Dictionary(article_words_list)
    # id2word_dictionary.filter_extremes(no_below=5, no_above=0.55, keep_n=5000)
    
    # for i, row in df.iterrows():
    #     article_word_tokens = row["tokens"]
    #     article_bag_of_words = id2word_dictionary.doc2bow(article_word_tokens)
    #     article_topic_distribution = lda_model.get_document_topics(article_bag_of_words, per_word_topics=False)
    #     max_tuple = max(article_topic_distribution, key=lambda x:x[1])
    #     max_topic_id = max_tuple[0]
    #     print(max_topic_id)
    #     filepath = row["filepath"]
    #     if "articles" not in topics_map[max_topic_id]:
    #         topics_map[max_topic_id]["articles"] = []
    #         print("adding articles")
    #     topics_map[max_topic_id]["articles"].append(filepath)

    # for key, value in topics_map.items():
    #     print("Topic: {} {}".format(key, value["Name"]))
    #     articles = value["articles"]
    #     for a in articles:
    #         print(" - {}".format(a))

        


    