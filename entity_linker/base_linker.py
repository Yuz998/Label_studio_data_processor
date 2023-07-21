import os, json
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EntityLinkerBase():
    def __init__(self, cache_dir, corpus_type) -> None:
        self.corpus_type = corpus_type
        self.cache_dir = cache_dir
        self.vectorizer_path = os.path.join(cache_dir, corpus_type, f"{corpus_type}_tf_idf_vectorizer.pickle")
        self.df_vector_path = os.path.join(cache_dir, corpus_type, f"{corpus_type}_tf_idf_vector.h5")
        self.read_tfidf_and_dict_config()
        logger.info(f'Corpus type: {self.corpus_type}')

    def read_tfidf_and_dict_config(self):
        with open('configs/TfidfVectorizer_configs.json', 'r', encoding='utf-8') as f:
            tfidf_config = json.load(f)
        self.tfidf_config = tfidf_config[self.corpus_type]
        with open('configs/dictionary_files.json', 'r', encoding='utf-8') as f:
            dict_path = json.load(f)
        self.corpus_dict_path = dict_path[self.corpus_type]
        
    def read_dictionary(self, delimiter=","):
        logger.info(f'Dictionary_file {self.corpus_dict_path} is start to be read by pandas')
        self.df_dict = pd.read_csv(self.corpus_dict_path, delimiter=delimiter, header=0, 
                         usecols=[0, 1], na_filter=False, names=["dbid", "text_name"])
        logger.info(f'Top 5 items are:\n{self.df_dict.head()}')
        
    def save_vectorizer(self, vectorizer):
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f, protocol=5)
        logger.info(f'Tfidf vectorizer is saved at {self.vectorizer_path}.')

    def save_tf_idf_dictionary(self, df):
        df.to_hdf(self.df_vector_path, key='df', mode='w')
        logger.info(f'Dictionary_file is saved at {self.df_vector_path}.')

    def load_vectorizer(self):
        with open(self.vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info(f'Tfidf vectorizer is loaded from {self.vectorizer_path}.')
        return vectorizer
    
    def load_tf_idf_dictionary(self):
        df = pd.read_hdf(self.df_vector_path, 'df')
        logger.info(f'Dictionary_file {self.df_vector_path} is read by pandas, top 5 items are:\n{df.head()}')
        logger.info(f'Dictionary_file {self.df_vector_path} shape:{df.shape}')
        return df.copy()
    
    def create_tfidf_vectorizer(self):
        logger.info(f'Creating Tf-idf dictionary and train vectorizer')
        self.read_dictionary()
        self.tf_idf_fit()
        self.init_index(self.df_dict.copy())
        logger.info('Data is successfully inserted into index! End.')
    
    def tf_idf_fit(self):
        if os.path.exists(self.vectorizer_path) and os.path.exists(self.df_vector_path):
            logger.info(f"Loading df_vector {self.df_vector_path}")
            self.df_dict = self.load_tf_idf_dictionary()
        else:
            ngram_range = (self.tfidf_config["ngram_range_left"], self.tfidf_config["ngram_range_right"])
            min_df, analyzer = self.tfidf_config["min_df"], self.tfidf_config["analyzer"]
            vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df)
            logger.info('Starting train tf-idf ... ')
            vectorizer.fit(self.df_dict.text_name)
            logger.info('Tfidf-fit finishes and starts to add vector into dataframe!')
            logger.info(f'Tfidf_config: {self.tfidf_config}')
            logger.info(f'Tfidf-fit feature number: {len(vectorizer.idf_)}')
            self.df_dict["tfidf_vector"] = self.df_dict.text_name.apply(lambda x: vectorizer.transform([x]))
            self.save_vectorizer(vectorizer)
            self.save_tf_idf_dictionary(self.df_dict.copy())
        return self.df_dict.copy()            

    def load_pretrained_data(self):
        raise NotImplementedError

    def init_index(self, df_dict):
        raise NotImplementedError

    def search_vector(self, entity_names):
        raise NotImplementedError