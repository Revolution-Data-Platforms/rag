# %%
"""RDP Modified Wrapper around Azure Cognitive Search."""
from typing import (
    Any,
    List,
    Optional,
    Tuple,
)
import requests, os
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore


def decode_base64(encoded_link):
    """
    Decodeing the base64 encoded link to the original link
    """
    cmd  = f'echo {encoded_link} | base64 --decode'
    stream = os.popen(cmd)
    output = stream.read()
    return output.strip()


class AzureSearch(VectorStore):
    """Wrapper around Azure Cognitive Search.
    
    Args:
        azure_search_endpoint: The endpoint of the Azure Cognitive Search service.
        azure_search_key: The key for the Azure Cognitive Search service.  
    """

    def __init__(
        self,
        azure_search_endpoint: str,
        azure_search_key: str,
    ):
        
        """Initialize with necessary components."""
        # Initialize base class
        self.azure_search_endpoint = azure_search_endpoint
        self.azure_search_key = azure_search_key

    def semantic_search_captions(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Return docs most similar semantics to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query
        """

        headers = {
        "Content-Type": "application/json",
        "api-key": self.azure_search_key
        }

        body = {
            "search": query,
            "queryType": "semantic",
            "searchFields": "content",
            "queryLanguage": "en-us",
            'top': k
        }

        response = requests.post(self.azure_search_endpoint, headers=headers, json=body)

        # Check if response is valid
        if response.status_code != 200:
            raise ValueError(
                f"Response returned with status code {response.status_code}: {response.text}"
            )
        results = response.json()["value"]

        # Convert results to Document objects
        docs = []
        for result in results:
            path = decode_base64(result['metadata_storage_path'])
            for caption in result['@search.captions']:
                docs.append(
                    Document(
                    page_content=caption['text'], 
                    metadata={
                        "file_path": path,
                        "search_score": result['@search.score'],
                        "search_reranker_score": result['@search.rerankerScore'],
                              }))
        return docs

    def semantic_search_contents(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """
        Return the whole docs most similar semantics to query.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        headers = {
        "Content-Type": "application/json",
        "api-key": self.azure_search_key
        }

        body = {
            "search": query,
            "queryType": "semantic",
            "searchFields": "content",
            "queryLanguage": "en-us",
            'top': k
        }

        response = requests.post(self.azure_search_endpoint, headers=headers, json=body)

        # Check if response is valid
        if response.status_code != 200:
            raise ValueError(
                f"Response returned with status code {response.status_code}: {response.text}"
            )
        results = response.json()["value"]

        # Convert results to Document objects
        docs = []
        for result in results:
            docs.append(
                    Document(
                    page_content=result['content'], 
                    metadata={
                        "file_path": decode_base64(result['metadata_storage_path']),
                        "search_score": result['@search.score'],
                        "search_reranker_score": result['@search.rerankerScore'],
                              }))                
        return docs

    def semantic_search_full_text(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """
        Return the whole docs most similar full search to query.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        headers = {
        "Content-Type": "application/json",
        "api-key": self.azure_search_key
        }

        body = {
            "search": query,
            "queryType": "full",
            "searchFields": "content",
            "queryLanguage": "en-us",
            'top': k
        }

        response = requests.post(self.azure_search_endpoint, headers=headers, json=body)

        # Check if response is valid
        if response.status_code != 200:
            raise ValueError(
                f"Response returned with status code {response.status_code}: {response.text}"
            )
        results = response.json()["value"]

        # Convert results to Document objects
        docs = []
        for result in results:
            docs.append(
                    Document(
                    page_content=result['content'], 
                    metadata={
                        "file_path": decode_base64(result['metadata_storage_path']),
                        "search_score": result['@search.score'],
                        "search_reranker_score": result['@search.rerankerScore'],
                              }))                
        return docs

    def add_texts(self, texts: List[str]) -> None:
        """Add texts to the vector store.

        Args:
            texts: List of texts to add to the vector store.
        """
        pass

    def from_texts(self, texts: List[str]) -> None:
        """Add texts to the vector store.

        Args:
            texts: List of texts to add to the vector store.
        """
        pass

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Add texts to the vector store.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            kwargs: Additional arguments to pass to the search function.
                    'search_type' The type of search to perform. Choose from "semantic_captions", "semantic_contents", "full".
                    Defaults to "semantic".
        """
        search_type = kwargs.get("search_type", "semantic")
        if search_type == 'semantic_captions':
            print('Searching in captions')
            return self.semantic_search_captions(query, k=k, **kwargs)
        elif search_type == 'semantic_contents':
            print('Searching in contents')
            return self.semantic_search_contents(query, k=k, **kwargs)
        elif search_type == 'full':
            print('Searching in full text')
            return self.semantic_search_full_text(query, k=k, **kwargs)
        else:
            return self.semantic_search_contents(query, k=k, **kwargs)
        

        