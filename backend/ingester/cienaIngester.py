from typing import List, Any
import os

from langchain.docstore.document import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import BSHTMLLoader

from baseIngester import baseIngester

class CienLoader(baseIngester):

    def load_doc(self, path: str) -> List[Document]:
        """
        Load a single document from a directory path

        Args:
            path (str): Path to the directory

        Returns:
            List[Document]: A list of documents
        """
        documents = []
        for file_path in tqdm(glob(os.path.join(path, "*"))):
            documents.extend(self.load_single_document(file_path))
        return documents
        loader = BSHTMLLoader("example_data/fake-content.html")
        data = loader.load()






