from typing import (List, 
                    Any)
from langchain.docstore.document import Document
import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader, 
    PDFPlumberLoader
)



class DocumentFactory:
    def __init__(self):
        self.Loader_Mapping = {
                                "csv": (CSVLoader, {}),
                                "docx": (Docx2txtLoader, {}),
                                "doc": (Docx2txtLoader, {}),
                                "enex": (EverNoteLoader, {}),
                                "epub": (UnstructuredEPubLoader, {}),
                                "html": (UnstructuredHTMLLoader, {}),
                                "md": (UnstructuredMarkdownLoader, {}),
                                "odt": (UnstructuredODTLoader, {}),
                                "pdf": (PDFPlumberLoader, {}),
                                "ppt": (UnstructuredPowerPointLoader, {}),
                                "pptx": (UnstructuredPowerPointLoader, {}),
                                "txt": (TextLoader, {"encoding": "utf8"}),
                                # Add more mappings for other file extensions and loaders as needed
                               }
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """
        Load a single document from a file path

        Args:
            file_path (str): Path to the document file

        Returns:
            List[Document]: A list of documents
        """
        try:
            file_extension = file_path.split(".")[-1]
            if file_extension in self.Loader_Mapping:
                loader_class, loader_args = self.Loader_Mapping[file_extension]
                loader = loader_class(file_path, **loader_args)
                return loader.load()
            
            raise ValueError(f"Unsupported file extension '{file_extension}'")
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            return []
    
    def load_documents_from_directory(self, source_dir: str, ignored_files: List[str] = []) -> List[Document]:
        """
        Loads all documents from the source documents directory, ignoring specified files

        Args:
            source_dir (str): Path to the source directory
            ignored_files (List[str], optional): List of file names to ignore. Defaults to [].

        Returns:
            List[Document]: A list of documents
        """

        all_files = []
        for ext in self.Loader_Mapping:
            all_files.extend(
                glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
        filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
        print(f"Found {len(filtered_files)} documents")
        with Pool(processes=os.cpu_count()) as pool:
            results = []
            with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
                for i, docs in enumerate(pool.imap_unordered(self.load_single_document, filtered_files)):
                    results.extend(docs)
                    pbar.update()
        return results
    
    def process_documents(self, documents: List [Document], chunk_size: int=1000, chunk_overlap: int=50) -> List[Document]:
        """
        Load documents and split in chunks

        Args:
            documents List[Document]: list of documents to be processed
            chunk_size [int]: size of the chunk. Defaults to 1000.
            chunk_overlap [int]: overlap between chunks. Defaults to 50.

        Returns:
            List[Document]: A list of documents
        """

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        return texts
    
    def to_FAISS(self, db_location, documents, embedding):
        """
        This method saves the documents in a FAISS vectorstore

        Args:
            db_location (str): The location of the FAISS vectorstore
            documents (list): The list of documents to be saved
            embedding (object): The embedding model to be used
        
        Returns:
            object: The FAISS vectorstore
        """

        # check if db_location does not exist. Create it and save the vectorstore
        if not os.path.exists(db_location):
            os.makedirs(db_location)
            vector_store = FAISS.from_documents(documents=documents, embedding=embedding)
            vector_store.save_local(db_location)

        # if db_location exists, load it and update it with the new documents
        else:
            vector_store = FAISS.load_local(db_location)
            vector_store.update(documents=documents, embedding=embedding)
            vector_store.save_local(db_location)
        return vector_store


    
    

    