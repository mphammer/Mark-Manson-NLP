from pprint import pprint
from unittest import result
import numpy as np
import re

# Gensim
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

# Topics I see:
# Relationship, love, partner, sex
# books, reading, writing, philosophy
# News, information, society 
# happiness, goals, values, meaning
# emotions, feelings, pain, mental health
# productivity, habits, goals, hours, attention 


def optimize_lda(article_words_list, num_topics_list, filter_words_in_less_then_n_docs_list, filter_words_in_more_than_n_percent_of_docs_list):
    results = {}
    iteration = 0
    for num_topics in num_topics_list:
        # num_articles = len(article_words_list)
        # articles_per_topic = num_articles / num_topics
        # less_than_docs_list = [articles_per_topic*2, articles_per_topic, articles_per_topic // 2, articles_per_topic // 4]
        for filter_words_in_less_then_n_docs in filter_words_in_less_then_n_docs_list:
            for filter_words_in_more_than_n_percent_of_docs in filter_words_in_more_than_n_percent_of_docs_list:
                coherence, topics = latent_dirishlet_allocation(article_words_list, num_topics, filter_words_in_less_then_n_docs, filter_words_in_more_than_n_percent_of_docs)
                print("Topics: {} | Above Docs: {:.2f} | Less Than %: {:.2f} | Coherence: {:.4f}".format(num_topics, filter_words_in_less_then_n_docs, filter_words_in_more_than_n_percent_of_docs, float(coherence)*-1.0))
                results[iteration] = {
                    "coherence": coherence,
                    "topics": topics,
                    "num_topics": num_topics,
                    "filter_words_in_less_then_n_docs": filter_words_in_less_then_n_docs,
                    "filter_words_in_more_than_n_percent_of_docs": filter_words_in_more_than_n_percent_of_docs
                }
                iteration += 1

    print("\n\n======================================")
    print("TOP RESULTS")
    print("======================================\n\n")

    num_results = min(15, len(results.items()))
    for i in range(num_results):
        max_coherence = 0.0
        max_results_key = 0
        for key, value  in results.items():
            if float(value["coherence"])*-1.0 > max_coherence:
                max_coherence = float(value["coherence"])*-1.0
                max_results_key = key
        max_result = results[max_results_key]
        print("Result: {} | Topics: {} | Above Docs: {:.2f} | Less Than %: {:.2f} | Coherence: {:.4f}".format(i+1, max_result["num_topics"], max_result["filter_words_in_less_then_n_docs"], max_result["filter_words_in_more_than_n_percent_of_docs"], max_result["coherence"]))
        # print("Topics:")
        for t in max_result["topics"]:
            t_topics = re.sub("0\.[0-9]+\*", "", t[1])
            t_topics = re.sub(" \+ ", "", t_topics)
            t_topics = re.sub("\"\"", ", ", t_topics)
            print("{}: {}".format(t[0], t_topics))
        # pprint(max_result["topics"])
        print()
        del results[max_results_key]


def latent_dirishlet_allocation(article_words_list, num_topics, filter_words_in_less_then_n_docs, filter_words_in_more_than_n_percent_of_docs):
    id2word_dictionary = corpora.Dictionary(article_words_list)
    #less than no_below documents (absolute number) or
    #more than no_above documents (fraction of total corpus size, not absolute number).
    #after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
    id2word_dictionary.filter_extremes(no_below=filter_words_in_less_then_n_docs, no_above=filter_words_in_more_than_n_percent_of_docs, keep_n=5000)

    bag_of_words_corpus = [id2word_dictionary.doc2bow(article_words) for article_words in article_words_list]
    # print(bag_of_words_corpus)
    

    # Build LDA model
    r = np.random.RandomState(42)
    lda_model = gensim.models.LdaMulticore(corpus=bag_of_words_corpus,
                                        id2word=id2word_dictionary,
                                        num_topics=num_topics,
                                        random_state=r,
                                        passes=25)

    # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics(num_topics=num_topics, num_words=10))
    topics = lda_model.show_topics(num_topics=num_topics, num_words=16)
    # doc_lda = lda_model[corpus]

    cm = CoherenceModel(model=lda_model, corpus=bag_of_words_corpus, coherence='u_mass')
    coherence = cm.get_coherence()  
    # print("Coherence: {}".format(coherence))
    # print()
    return (lda_model, coherence, topics)
