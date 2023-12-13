
import os
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

class baseEmbedder(object):

    def __init__(self, model_name: str= "all-MiniLM-L6-v2"):
        self.embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
        self.model_name = model_name

    def embed(self, data):
        self.embedding_function.embed(data)
    
    def embedBatch(self, data):
        raise NotImplementedError("Batch embedding method not implemented")
    
    def embedBatchGenerator(self, data):
        raise NotImplementedError("Batch embedding generator method not implemented")
    