
import os
from langchain.vectorstores import FAISS


class baseEmbedder(object):

    def __init__(self, config):
        self.config = config

    def embed(self, data):
        raise NotImplementedError("Embedding method not implemented")
    
    def embedBatch(self, data):
        raise NotImplementedError("Batch embedding method not implemented")
    
    def embedBatchGenerator(self, data):
        raise NotImplementedError("Batch embedding generator method not implemented")
    