# %%
# setting path
import sys
sys.path.append('../../')

from core.DocumentFactory import DocumentFactory
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

if __name__ == '__main__':

    load_dotenv('./data_ingestion.env')

    # Gettting environment variables
    drop_zone_location   = os.environ.get('drop_zone_location')
    vectorstore_location = os.environ.get('vectorstore_location')
    vectorstore_name     = os.environ.get('vectorstore_name')
    embedding_model_name = os.environ.get('embedding_model_name')

    Factory = DocumentFactory()

    # Load documents in drop zone
    docs = Factory.load_documents_from_directory(drop_zone_location)
    # Process documents
    texts = Factory.process_documents(docs)
    embedding = HuggingFaceInstructEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cuda'})
    # Save to FAISS vectorstore
    Factory.to_FAISS(db_location=f"{vectorstore_location}/{vectorstore_name}", 
                     documents=texts,
                     embedding=embedding)



# %%
