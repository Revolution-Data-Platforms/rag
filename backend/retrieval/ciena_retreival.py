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
        self.db = kwargs['db']
        # self.key_db = BM25Retriever.from_documents(kwargs['db'], search_kwargs={"k": self.k, "threshold": self.threshold})

    def get_semantic_res(self, query):
        """Retrieve semantic results from Ciena database."""
        retriever = self.db.as_retriever(search_kwargs={'k': self.k, 'threshold': self.threshold})
        semantic_res = retriever.get_relevant_documents(query= query)
        # import pdb; pdb.set_trace()
        return semantic_res

    def get_keyword_res(self, query):
        """Retrieve keyword results from Ciena database."""
        key_word_res = self.key_db.get_relevant_documents(query)
        return key_word_res

    def get_res(self, query):
        """Retrieve Hybrid search results from Ciena database."""
        semantic_res = self.get_semantic_res(query)
        # keyword_res = self.get_keyword_res(query) if self.hybrid else []
        final_res = semantic_res #+ keyword_res
        return final_res

    def get_context(self, headers):
        context = []
        res = self.db.get(where={"header": {"$in": headers}})

        source = {}
        for item in res['metadatas']:
            source['page_number'] = item['page_number']
            source['source'] = item['source']
            break
        
        for i, item in enumerate(res['metadatas']):
            cur_header = item['header']
            table_dir = os.path.dirname(item["source"])

            if 'Table' in item["type"] and item['table_path'] != '':
                table_path = os.path.join(table_dir, item['table_path'])
                context.append(md_table(table_path))
            else:
                context.append(res['documents'][i])
            
            if i < len(res['metadatas']) - 1:
                next_header = res['metadatas'][i+1]['header']
                if next_header != cur_header:
                    context.append('\n\n')
        return context, source

