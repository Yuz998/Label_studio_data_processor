import os, json
import faiss
import numpy as np
from .base_linker import EntityLinkerBase
import torch

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Faiss2EntityLinker(EntityLinkerBase):
    def __init__(self, cache_dir, corpus_type, enable_gpu_linker, device) -> None:
        super().__init__(cache_dir, corpus_type)
        self.device = device
        self.config = self.read_faiss_config()
        self.nprobe = self.config["nprobe"]
        self.use_gpu = 0
        if enable_gpu_linker and torch.cuda.is_available():
            self.use_gpu = 1
        cache_dir = os.path.join(cache_dir, corpus_type)
        os.makedirs(cache_dir, exist_ok=True)
        self.index_path = os.path.join(cache_dir, f"{corpus_type}_faiss_index.dat")
        self.load_pretrained_data()
    
    def read_faiss_config(self):
        with open('configs/faiss_configs.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_config(self):
        with open('configs/faiss_configs.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
            
    def init_faiss(self, dim_size, nlist=1024):
        quantizer = faiss.IndexFlatL2(dim_size)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, dim_size, nlist, faiss.METRIC_L2)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        return self.faiss_index
    
    def init_index(self, df):
        logger.info(f'Pandas for faiss insert, top 5 items are:\n{df.head()}')
        dim_size = df.tfidf_vector[0].toarray().shape[1]
        logger.info(f'vector_for_insert dim_size: {dim_size}')
        logger.info(f'df_for_insert shape: {df.shape}')
        logger.info(f'df_for_insert column names: {df.columns}')
        current_config = self.config[self.corpus_type]
        current_config['vector_for_insert_dim_size'] = dim_size
        current_config['df_for_insert_shape'] = df.shape
        current_config['df_for_insert_column_names'] = list(df.columns)
        self.save_config()
        
        self.init_faiss(dim_size)
        target_vectors = np.array(df["tfidf_vector"].apply(lambda x: x.toarray()[0]).tolist()).astype(np.float32)
        logger.info('Finish to convert spare to dense vector in dataframe.')
        assert not self.faiss_index.is_trained
        logger.info(f'Starting train ...')
        self.faiss_index.train(target_vectors)
        assert self.faiss_index.is_trained
        logger.info(f'Starting add ...')
        self.faiss_index.add(target_vectors)
        
        if self.use_gpu:
            self.faiss_index = faiss.index_gpu_to_cpu(self.index_path)
        faiss.write_index(self.faiss_index, self.index_path)
        logger.info(f'Save faiss index at {self.index_path}')

    def load_pretrained_data(self):
        logger.info(f"Load faiss index...")
        if os.path.exists(self.index_path):
            self.faiss_index = faiss.read_index(self.index_path)
        else:
            logger.info(f"Creating {self.corpus_type} faiss index...")
            self.create_tfidf_vectorizer()
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, int(self.device), self.faiss_index)
        self.faiss_index.nprobe = self.nprobe
        logger.info(f'Faiss index.ntotal {self.faiss_index.ntotal}')
        logger.info(f'Faiss index.d {self.faiss_index.d}')
        current_config = self.config[self.corpus_type]
        item_len, vec_dim = current_config["df_for_insert_shape"][0], current_config["vector_for_insert_dim_size"]
        assert self.faiss_index.ntotal == item_len, f"Index sample nums {self.index.ntotal} != Items length {item_len}"
        assert self.faiss_index.d == vec_dim, f"Index dim {self.faiss_index.d} != Vecs dim {vec_dim}"
        assert self.faiss_index.is_trained, "Index dose not trained"
        self.df_dict = self.load_tf_idf_dictionary()
        self.vectorizer = self.load_vectorizer()
    
    def search_vector(self, entity_names):
        ent_vectors = self.vectorizer.transform(entity_names).toarray().astype(np.float32)
        self.distances, self.faiss_indexes = self.faiss_index.search(ent_vectors, 1)
        linked_entity_names = []
        for distance, search_item, entity_name in zip(self.distances, self.faiss_indexes, entity_names):
            top_id = search_item[0]
            linked_entity_name = self.df_dict.text_name.iloc[top_id]
            entity_id = self.df_dict.dbid.iloc[top_id]
            top_distance = float(distance[0])
            extra_info = {}
            extra_info["dbid"] = entity_id
            top_distance = float(distance[0])
            linked_entity_names.append((linked_entity_name, top_distance, extra_info))
        return linked_entity_names
            

        
