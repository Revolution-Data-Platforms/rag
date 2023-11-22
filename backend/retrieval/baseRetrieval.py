import re
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


class baseRetrieval(object):
    def __init__(self, vectore_store, embedding, k=5, AzSearch_search_type='semantic_contents', clean_text = True) -> None:

        """
        This class is AIO class for DataMind. It uses existing vectorstore and embedding to perform search and generate prompt for the generation

        Args:
            vectore_store (vectorstore.VectorStore): The vectorstore to be used for the search
            embedding (embeddings.Embedding): The embedding to be used for the search
            k (int, optional): The number of documents to be retrieved. Defaults to 5.
            AzSearch_search_type (str, optional): ONLY USED For Azurre Search. The search type to be used for the search. Defaults to 'semantic_contents'.
            clean_text (bool, optional): Whether to clean the text or not. Defaults to True.        
        """
        self.vectorstore = vectore_store
        self.embedding = embedding
        self.K = k
        self.AzSearch_search_type = AzSearch_search_type
        self.clean_text = clean_text
        

    def _clean_text(self, text):
        """
        This method cleans the text by removing the special characters and extra spaces

        Args:
            text (str): The text to be cleaned

        Returns:
            str: The cleaned text
        """
        text = re.sub(r'#', '', text)
        text = re.sub(r'\*', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\t', '', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def _chunker(self, documents, chunk_size=1200, chunk_overlap=100, length_function=len, separators=["\n\n\n", "\n\n", "\n", ". ", ' ', ""]):
        """
        This method chunks the documents into smaller chunks

        Args:
            documents (list): The list of documents to be chunked
            chunk_size (int, optional): The size of the chunk. Defaults to 1200.
            chunk_overlap (int, optional): The overlap between the chunks. Defaults to 100.
            length_function (function, optional): The function to be used to calculate the length of the chunk. Defaults to len.
            separators (list, optional): The list of separators to be used for the chunking. Defaults to ["\n\n\n", "\n\n", "\n", ". ", ' ', ""].

        Returns:
            list: The list of chunked documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
                                        chunk_size=chunk_size, 
                                        chunk_overlap=chunk_overlap, 
                                        length_function=length_function,
                                        separators=separators)
        return text_splitter.split_documents(documents)

    def search_in_document(self, Ask_question):
        """
        This method searches the question in the document and returns the context and sources

        Args:
            Ask_question (str): The question to be searched

        Returns:
            list: The list of context
            list: The list of sources
        """

        related_documents = self.vectorstore.similarity_search(Ask_question, k=self.K, search_type=self.AzSearch_search_type)
        split_documents = self._chunker(related_documents)
        Chunck_indexer = FAISS.from_documents(split_documents, self.embedding)
        related_chunks = Chunck_indexer.similarity_search(Ask_question, k=3)
        context, sources = [], []
        for idx, each_response in enumerate(related_chunks):
            sources.append((each_response.metadata['file_path'], each_response.metadata['page']))
            if self.clean_text:
                context.append({f"context-{idx+1}":self._clean_text(each_response.page_content)})
            else:
                context.append({f"context-{idx+1}":each_response.page_content})
        return context, sources
        
    def related_documents(self, Ask_question):
        """
        This method searches the question in the document and returns the related documents

        Args:
            Ask_question (str): The question to be searched

        Returns:
            list: The list of related documents
        """

        related_documents = self.vectorstore.similarity_search(Ask_question, k=self.K, search_type=self.AzSearch_search_type)
        split_documents = self._chunker(related_documents)
        Chunck_indexer = FAISS.from_documents(split_documents, self.embedding)
        related_chunks = Chunck_indexer.similarity_search(Ask_question, k=3)
        return related_chunks   

    def generate_prompt_with_context_and_sources(self, system_prompt, question, assistant_token = "FINAL ANSWER: "):
        """
        This function generates the final prompt to be used for the generation. Context is passed as a json object.

        Args:
            system_prompt (str): The system prompt to be used for the generation
            question (str): The question to be answered
            assistant_token (str): The assistant token to be used for the generation

        Returns:
            str: The final prompt to be used for the generation
        """

        # get context data 
        context_text_as_json, sources = self.search_in_document(question)

        # form the final prompt
        prompt = system_prompt + "\n\n" + "Question: {question}\n\n" + "Extracted parts: {context_data}\n\n" + assistant_token
    

        return prompt.format(question = question, context_data=context_text_as_json).strip(), sources
