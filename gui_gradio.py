import gradio as gr
import os
from backend.llm.baseLLM import Remote_LLM 
# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.
from backend.retrieval.ciena_retreival import CienaRetrieval
from backend.embedder.baseEmbedder import baseEmbedder
from backend.retrieval.utils import *
from backend.retrieval.rereanker import Reranker
from langchain.document_loaders import JSONLoader

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


_executor = ThreadPoolExecutor(1)



embedding_function = baseEmbedder().embedding_function
retriael_kwargs = {
    "threshold": "0.8",
    "k": 20,
    "embedder": embedding_function,
    "hybrid": True
}
ciena_retreival = CienaRetrieval(**retriael_kwargs)
reranker = Reranker()

def load_db():
    """Load Ciena database."""
    
    dir = './output/'
    loaded_data = []
    for r, d, f in os.walk(dir):
        
        for file in f: 
            # if "10Aug-BP_Engineering_guide" in file or "Blue_Planet_MLA_Cloud_Deployment_Guide" in file:
            if '.json' in file and file != 'structuredData.json':
                dir_ = file.split('.')[0] + '.pdf'
                file_name = os.path.join(dir, dir_, file)
                print(file_name)
                try:
                    loader = JSONLoader(
                        file_path=file_name,
                        jq_schema='.[].content[]',
                        content_key="text", 
                        metadata_func=metadata_func)

                    loaded_data.extend(loader.load())
                    print(f"Successfully loaded file {file_name}")
                except Exception as e:
                    # import pdb; pdb.set_trace()
                    print(f"error in loading  file {file_name}")
                    print(e)

    return loaded_data

def clean(docs):
    loaded_data = filter_empty(docs)
    loaded_data = filter_redundant(loaded_data)
    loaded_data = exclude_toc(loaded_data)
    return loaded_data


def get_relevant_docs(query, docs):
    """Get relevant documents from Ciena database."""
    if len(query) == 0 or len(docs) == 0:
        return []
    ciena_retrieval = CienaRetrieval(**retriael_kwargs)
    relevant_docs = ciena_retrieval.get_res(query, docs)
    reranked_res = reranker.rerank(query, relevant_docs)
    return reranked_res

def get_context(docs, headers):
    if len(headers) == 0:
        return [], []
    context, sources = ciena_retreival.get_context(docs, headers)
    return context, sources

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

def bot(history):
    response = "**That's cool!**"
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history



def main_get_src_ctx(message, seconds):
    query = message
    relevant_docs = get_relevant_docs(query, cleaned_db)
    rel_headers = relevant_headers(relevant_docs)
    rel_headers = [x for x in rel_headers if x != 'Table of Contents ']
    context, sources = get_context(loaded_db, rel_headers)
    return context, sources

def gt_llm_answer(question, ctx, src):
    endpoint = " http://0.0.0.0:8000/answer"
    LLM_kwargs={'max_new_tokens': 500, 'temperature': 0.5}

    llm = Remote_LLM(
        endpoint="http://0.0.0.0:8000/answer",
        generation_config=LLM_kwargs
    )
    ctx = ctx[len(ctx) // 2:]
    if len(ctx) > 2000: 
        ctx = ctx[:2000]
    prompt = f"""
    You are a powerful AI asistant that answers only based on the given contex. If the context is not enough, you can ask for more information.
    Given the following context {ctx}, answer the following question: {question}
    """

    answer = llm(prompt)
    return answer, src


def slow_echo(message, history):
    ctx, src = main_get_src_ctx(message, 3)
    # convert list ctx to string
    ctx =' '.join(ctx)
    answer, src = gt_llm_answer(message, ctx, src)
    return answer #

def main():
    global loaded_db
    global cleaned_db
    loaded_db = load_db()
    import pdb; pdb.set_trace()
    cleaned_db = clean(loaded_db)
    gr.ChatInterface(
        slow_echo,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
        title="BP Chatbot",
        description="Ask Yes Man any question",
        theme="soft",
        examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
        cache_examples=True,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
    ).launch(server_port= 8888)

if __name__ == "__main__":
    main()
