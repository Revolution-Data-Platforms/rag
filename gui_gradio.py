import gradio as gr
import os
from backend.llm.baseLLM import Remote_LLM 
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

def load_db(dir= './output/'):
    """Load Ciena database."""
    
    loaded_data = []
    for r, d, f in os.walk(dir):
        
        for file in f:
            if '.json' in file and file != 'structuredData.json':
                file_name = os.path.join(r, file)
                try:
                    loader = JSONLoader(
                        file_path=file_name,
                        jq_schema='.[].content[]',
                        content_key="text",
                        text_content=False,
                        metadata_func=metadata_func)

                    loaded_data.extend(loader.load())
                    print(f"Successfully loaded file {file_name}")
                except Exception as e:
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
    LLM_kwargs={'max_new_tokens': 1500, 'temperature': 0.4}

    llm = Remote_LLM(
        endpoint=endpoint,
        generation_config=LLM_kwargs
    )
    full_prompt = f"""\
        <|system|> Given a part of a lengthy markdown document, answer the following question: `{question}`. Please, follow the same format as the source document given. </s>
        <|user|>
        please ONLY respond with: {{not_found_response}}, if the context does not provide the answer </s>
        CONTEXT: {ctx} 

        <|assistant|> """

    answer = llm(full_prompt)
    return answer, src


def slow_echo(message, history):
    ctx, src = main_get_src_ctx(message, 3)
    ctx = remove_duplicates_preserve_order(ctx)
    ctx = '\n'.join(ctx)
    answer, src = gt_llm_answer(message, ctx, src)
    bot_response = answer.split('<|assistant|>')[1].split('</s>')[0]
    # with open('response.txt', 'w') as f:
    #     f.write(bot_response)
        
    return bot_response #


loaded_db = load_db()
cleaned_db = clean(loaded_db)
def main():
    
    gr.ChatInterface(
        slow_echo,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Ask Me any question related to BP Docs", container=False, scale=7),
        title="BP Chatbot",
        description="Ask Me any question related to BP Docs",
        theme="soft",
        examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
        cache_examples=True,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
    ).launch(server_port= 8888)

if __name__ == "__main__":
    main()
