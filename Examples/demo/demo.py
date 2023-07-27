# setting path
import sys
sys.path.append('../../')

from UI.streamlit.UI import streamlit_UI, streamlit_QA
from core.DataMind import DataMind
import os
from dotenv import load_dotenv


 


if __name__ == '__main__':
    
    load_dotenv('./demo.env')
    embedding_model = os.environ.get('embedding_model')
    vectorstore_location = os.environ.get('vectorestore_location')
    LLM_endpoint = os.environ.get('LLM_endpoint')

    print(embedding_model)
    print(vectorstore_location)
    print(LLM_endpoint)

    embeddings, vectorstore, LLM_instance = streamlit_UI(embedding_model=embedding_model,
    vectorstore_location=vectorstore_location,
    LLM_endpoint=LLM_endpoint, 
    LLM_kwargs={'max_new_tokens': 500, 'temperature': 0.5})
    
    context_generator = DataMind(vectore_store=vectorstore,
                                 embedding=embeddings,
                                 k=3)
    
    init_prompt ="""Given the following extracted parts from a long document and a question. Create a final answer for the question. If the answer is not found in the extracted parts, then answer with 'I do not know. There is not enough context.' """

    assistant_token = "### Assistant:"


    streamlit_QA(context_generator, LLM_instance, system_prompt=init_prompt, assistant_token=assistant_token)
    