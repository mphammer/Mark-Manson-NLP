from pprint import pprint
from unittest import result
import numpy as np
import re

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import datapath

# def optimize_lda(article_words_list, num_topics_list, filter_words_in_less_then_n_docs_list, filter_words_in_more_than_n_percent_of_docs_list):
#     '''
#     This is a function I hacked together to do a Grid Search before I knew what a Grid Search was.
#     TODO: I think it's broken and needs refactoring
#     '''
#     results = {}
#     iteration = 0
#     for num_topics in num_topics_list:
#         # num_articles = len(article_words_list)
#         # articles_per_topic = num_articles / num_topics
#         # less_than_docs_list = [articles_per_topic*2, articles_per_topic, articles_per_topic // 2, articles_per_topic // 4]
#         for filter_words_in_less_then_n_docs in filter_words_in_less_then_n_docs_list:
#             for filter_words_in_more_than_n_percent_of_docs in filter_words_in_more_than_n_percent_of_docs_list:
#                 coherence, topics = latent_dirishlet_allocation(article_words_list, num_topics, filter_words_in_less_then_n_docs, filter_words_in_more_than_n_percent_of_docs)
#                 print("Topics: {} | Above Docs: {:.2f} | Less Than %: {:.2f} | Coherence: {:.4f}".format(num_topics, filter_words_in_less_then_n_docs, filter_words_in_more_than_n_percent_of_docs, float(coherence)*-1.0))
#                 results[iteration] = {
#                     "coherence": coherence,
#                     "topics": topics,
#                     "num_topics": num_topics,
#                     "filter_words_in_less_then_n_docs": filter_words_in_less_then_n_docs,
#                     "filter_words_in_more_than_n_percent_of_docs": filter_words_in_more_than_n_percent_of_docs
#                 }
#                 iteration += 1

#     print("\n\n======================================")
#     print("TOP RESULTS")
#     print("======================================\n\n")

#     num_results = min(15, len(results.items()))
#     for i in range(num_results):
#         max_coherence = 0.0
#         max_results_key = 0
#         for key, value  in results.items():
#             if float(value["coherence"])*-1.0 > max_coherence:
#                 max_coherence = float(value["coherence"])*-1.0
#                 max_results_key = key
#         max_result = results[max_results_key]
#         print("Result: {} | Topics: {} | Above Docs: {:.2f} | Less Than %: {:.2f} | Coherence: {:.4f}".format(i+1, max_result["num_topics"], max_result["filter_words_in_less_then_n_docs"], max_result["filter_words_in_more_than_n_percent_of_docs"], max_result["coherence"]))
#         # print("Topics:")
#         for t in max_result["topics"]:
#             t_topics = re.sub("0\.[0-9]+\*", "", t[1])
#             t_topics = re.sub(" \+ ", "", t_topics)
#             t_topics = re.sub("\"\"", ", ", t_topics)
#             print("{}: {}".format(t[0], t_topics))
#         # pprint(max_result["topics"])
#         print()
#         del results[max_results_key]

class LatentDirishletAllocation:
    def __init__(self):
        self.id2word_dictionary = None
        self.list_of_documents_as_tokens = None
        self.bag_of_words_corpus = None
        self.model = None

        self.saved_model_datapath = datapath("model")

    def create_dictionary(self, list_of_documents_as_tokens, filter_words_in_less_then_n_docs, filter_words_in_more_than_n_percent_of_docs):
        self.list_of_documents_as_tokens = list_of_documents_as_tokens
        self.id2word_dictionary = corpora.Dictionary(list_of_documents_as_tokens)
        self.id2word_dictionary.filter_extremes(no_below=filter_words_in_less_then_n_docs, no_above=filter_words_in_more_than_n_percent_of_docs, keep_n=5000)
        return self.id2word_dictionary

    def create_bag_of_words_corpus(self):
        if self.id2word_dictionary == None:
            print("ERROR: Please create_dictionary first")
            return

        self.bag_of_words_corpus = [self.id2word_dictionary.doc2bow(doc_tokens) for doc_tokens in self.list_of_documents_as_tokens]
        return self.bag_of_words_corpus

    def create_model(self, num_topics):
        if self.id2word_dictionary == None:
            print("ERROR: Please create id2word dictionary first")
            return 
        if self.bag_of_words_corpus == None: 
            print("ERROR: Please create the bag of workds corpus first")
            return 

        r = np.random.RandomState(42)
        self.model = gensim.models.LdaMulticore(corpus=self.bag_of_words_corpus,
            id2word=self.id2word_dictionary,
            num_topics=num_topics,
            random_state=r,
            passes=25)
        return self.model
    
    def load_model(self):
        self.model = gensim.models.LdaMulticore.load(self.saved_model_datapath)
        return self.model

    def save_model(self):
        self.model.save(self.saved_model_datapath)

    def get_coherence_score(self):
        if self.model == None:
            print("ERROR: Please create the model first")
            return 
        if self.bag_of_words_corpus == None:
            print("ERROR: Please create the bag of workds corpus first")
            return 

        cm = CoherenceModel(model=self.model, corpus=self.bag_of_words_corpus, coherence='u_mass')
        return cm.get_coherence()  