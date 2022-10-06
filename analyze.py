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

    lda = LatentDirichletAllocation.LatentDirishletAllocation()

    num_topics = 6
    no_below = 25
    no_above = 0.50
    id2word_dictionary = lda.create_dictionary(article_words_list, no_below, no_above)
    lda.create_bag_of_words_corpus()
    # lda_model = lda.create_model(num_topics)
    # lda.save_model()
    lda_model = lda.load_model()
    coherence = lda.get_coherence_score()

    topics_data = {}
    topicid_word_tuples = lda_model.show_topics(num_topics=num_topics, num_words=12, formatted=False)
    for topicid_word_tuple in topicid_word_tuples:
        topic_id = topicid_word_tuple[0]
        topic_words_probability_tuples = topicid_word_tuple[1]
        words_string = " ".join([w[0] for w in topic_words_probability_tuples])
        print("{}: {}".format(topic_id, words_string))
        topics_data[topic_id] = {
            "topic_words_probability_tuples": topic_words_probability_tuples,
            "clean_words": words_string
        }
    
    # After manually looking at the topics
    topics_data[0]["Name"] = "World - News, Society, Culture"
    topics_data[1]["Name"] = "Brain - Reading, Learning, Wisdom"
    topics_data[2]["Name"] = "Sex and Dating Women"
    topics_data[3]["Name"] = "Dealing with Bad Emotions"
    topics_data[4]["Name"] = "Pursuit of Happiness"
    topics_data[5]["Name"] = "Relationship Advice"

    for i, row in df.iterrows():
        article_word_tokens = row["tokens"]
        article_bag_of_words = id2word_dictionary.doc2bow(article_word_tokens)
        article_topic_distribution_tuples = lda_model.get_document_topics(article_bag_of_words, per_word_topics=False)
        
        # Get the TopicID that has the greatest probability
        max_tuple = max(article_topic_distribution_tuples, key=lambda x:x[1])
        max_topic_id = max_tuple[0]
        
        # Add the file to that topic
        filepath = row["filepath"]
        if "articles" not in topics_data[max_topic_id]:
            topics_data[max_topic_id]["articles"] = []
        topics_data[max_topic_id]["articles"].append(filepath)

    # 
    for key, value in topics_data.items():
        print("Topic: {} {}".format(key, value["clean_words"]))
        articles = value["articles"]
        for a in articles:
            print(" - {}".format(a))
    