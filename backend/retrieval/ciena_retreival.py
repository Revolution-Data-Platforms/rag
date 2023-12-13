import os
from .utils import md_table
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever

class CienaRetrieval:
    """Class for retrieving data from Ciena database."""

    def __init__(self, **kwargs):
        """Initialize CienaRetrieval object."""
        self.threshold = kwargs['threshold']
        self.k = kwargs['k']
        self.embedder = kwargs['embedder']
        self.hybrid = kwargs['hybrid']

    def get_semantic_res(self, query, docs):
        """Retrieve semantic results from Ciena database."""
        db = Chroma.from_documents(docs, self.embedder)
        retriever = db.as_retriever(search_kwargs={'k': self.k, 'threshold': self.threshold})
        semantic_res = retriever.get_relevant_documents(query= query)
        return semantic_res

    def get_keyword_res(self, query, docs):
        """Retrieve keyword results from Ciena database."""
        key_word_retriever = BM25Retriever.from_documents(docs, search_kwargs={"k": self.k, "threshold": self.threshold})
        key_word_res = key_word_retriever.get_relevant_documents(query)
        return key_word_res

    def get_res(self, query, docs):
        """Retrieve Hybrid search results from Ciena database."""
        semantic_res = self.get_semantic_res(query, docs)
        keyword_res = self.get_keyword_res(query, docs) if self.hybrid else []
        final_res = semantic_res + keyword_res
        return final_res

    def get_context(self, docs):
        db = Chroma.from_documents(docs, self.embedder)
        context = []
        import pdb;pdb.set_trace()
        for doc in docs:
            base_name = os.path.basename(doc.metadata["source"])
            table_dir = os.path.dirname(doc.metadata["source"])

            res = db.get(where={"header": doc.metadata["header"]})

            for i, item in enumerate(res['metadatas']):
                if 'Table'in item["type"] and item['table_path'] != '':
                    context.append(md_table(os.path.join(table_dir, item['table_path'])))
                else:
                    context.append(res['documents'][i])
        return context

