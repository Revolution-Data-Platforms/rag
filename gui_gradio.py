import gradio as gr
import os
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
from backend.llm.cienaLLM import Remote_LLM 
from backend.retrieval.ciena_retreival import CienaRetrieval
from backend.embedder.baseEmbedder import baseEmbedder
from backend.retrieval.utils import *
from backend.retrieval.rereanker import Reranker
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import Chroma

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

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


def get_relevant_docs(query):
    """Get relevant documents from Ciena database."""
    if len(query) == 0:
        return []
    # ciena_retrieval = CienaRetrieval(**retrieval_kwargs)
    relevant_docs = ciena_retreival.get_res(query)
    reranked_res = reranker.rerank(query, relevant_docs)
    return reranked_res

def get_context(headers):
    if len(headers) == 0:
        return [], []
    context, sources = ciena_retreival.get_context(headers)
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



def main_get_src_ctx(message):
    query = message
    import pdb;pdb.set_trace()
    relevant_docs = get_relevant_docs(query)
    rel_headers = relevant_headers(relevant_docs)
    context, sources = get_context(rel_headers)
    return context, sources

def gt_llm_answer(question, ctx, src):
    # endpoint = " http://0.0.0.0:8000/answer"
    # LLM_kwargs={'max_new_tokens': 1500, 'temperature': 0.4}

    llm = Remote_LLM()
#     full_prompt = f"""\
# <|system|> Given a part of a lengthy markdown document, answer the following question: `{question}`. Please, follow the same format as the source document given. </s>
# <|user|>
# please ONLY respond with: {{not_found_response}}, if the context does not provide the answer </s>
# CONTEXT: {ctx} 

# <|assistant|> """

    answer = llm.generate(prompt= question, ctx= ctx)
    return answer, src


def slow_echo(message, history):
    ctx, src = main_get_src_ctx(message)
    ctx = remove_duplicates_preserve_order(ctx)
    ctx = '\n'.join(ctx)
    print(ctx)
    # answer, src = gt_llm_answer(message, ctx, src)
    # bot_response = answer.replace('_x000D_', ' ')
    # bot_response = bot_response.replace('x000D', ' ')
    # source = Path(src['source']).stem
    # ref_md = "# Source: " + source + " \npage number: " + str(src['page_number']) + '\n\n'
    # final_answer = bot_response + '\n\n' + ref_md
    # print(ctx)
    # return final_answer #

embedding_function = baseEmbedder().embedding_function
vectordb = Chroma(persist_directory="./db-uaa", embedding_function=embedding_function)
retrieval_kwargs = {
    "threshold": "0.8",
    "k": 20,
    "embedder": embedding_function,
    "hybrid": True,
    "db": vectordb
}
ciena_retreival = CienaRetrieval(**retrieval_kwargs)
reranker = Reranker()

def main():

    ## UAA_AI_SDK_Handbook_ 230864_1-0.pdf
    # slow_echo("What is the AI SDK?", None)
    # slow_echo("How do I add an AI App in the AI SDK?", None)
    # slow_echo("How do I use the AI Apps Manager?", None)

    # ########### BP Engineering Guide
    slow_echo("Give me a table for ciena's BPO Runtime License", None)
    # slow_echo("In the BPO Runtime Licenses table, what is the part number for MCO?", None)
    # slow_echo("Where can I deploy Blue Planet platform and solutions?", None)
    # slow_echo(" What are Supported OS for cloud deployment using Kubernetes platform", None)
    # slow_echo("What are the Server requirements for deploying SOO/MDSO/MCO on AWS on multiple hosts?", None)

    # gr.ChatInterface(
    #     slow_echo,
    #     chatbot=gr.Chatbot(height=500),
    #     textbox=gr.Textbox(placeholder="Ask Me any question related to BP UAA Docs", container=False, scale=7),
    #     title="BP Chatbot",
    #     description="Ask Me any question related to BP UAA Docs",
    #     theme="soft",
    #     # examples=["hi"],
    #     cache_examples=False,
    #     retry_btn=None,
    #     undo_btn="Delete Previous",
    #     clear_btn="Clear"
    # ).launch(server_port= 8888, server_name= "0.0.0.0")

if __name__ == "__main__":
    main()
