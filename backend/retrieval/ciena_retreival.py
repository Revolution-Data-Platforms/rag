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
        self.db = Chroma.from_documents(kwargs['db'], self.embedder)
        self.key_db = BM25Retriever.from_documents(kwargs['db'], search_kwargs={"k": self.k, "threshold": self.threshold})

    def get_semantic_res(self, query, docs):
        """Retrieve semantic results from Ciena database."""
        db = self.db
        retriever = db.as_retriever(search_kwargs={'k': self.k, 'threshold': self.threshold})
        semantic_res = retriever.get_relevant_documents(query= query)
        return semantic_res

    def get_keyword_res(self, query, docs):
        """Retrieve keyword results from Ciena database."""
        key_word_retriever = self.key_db
        key_word_res = key_word_retriever.get_relevant_documents(query)
        return key_word_res

    def get_res(self, query, docs):
        """Retrieve Hybrid search results from Ciena database."""
        semantic_res = self.get_semantic_res(query, docs)
        keyword_res = self.get_keyword_res(query, docs) if self.hybrid else []
        final_res = semantic_res + keyword_res
        return final_res

    def get_context(self, docs, headers):
        db = self.db
        context = []
        res = db.get(where={"header": {"$in": headers}})

        sources = {}

        for i, item in enumerate(res['metadatas']):
            cur_header = item['header']
            table_dir = os.path.dirname(item["source"])

            data_dir = './data'
            pdf_dir = os.path.basename(item["source"]).split('.')[0]
            pdf_name = pdf_dir + '.pdf'
            pdf_path = os.path.join(data_dir, pdf_dir, pdf_name)

            if pdf_path not in sources:
                sources[pdf_name] = pdf_path

            if 'Table' in item["type"] and item['table_path'] != '':
                table_path = os.path.join(table_dir, item['table_path'])
                context.append(md_table(table_path))
            else:
                context.append(res['documents'][i])
            
            if i < len(res['metadatas']) - 1:
                next_header = res['metadatas'][i+1]['header']
                if next_header != cur_header:
                    context.append('\n\n')
        return context, sources

