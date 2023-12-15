import json
import os
import logging
import requests
import openai
import os, uuid, tempfile

from flask import Flask, Response, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv('main.env')


app = Flask(__name__, static_folder="static")

# Static Files
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/favicon.ico")
def favicon():
    return app.send_static_file('favicon.ico')

@app.route("/assets/<path:path>")
def assets(path):
    return send_from_directory("static/assets", path)


def format_as_ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"



@app.route("/conversation", methods=["GET", "POST"])
def conversation():
    request_body = request.json
    return conversation_internal(request_body)

def conversation_internal(request_body):
    try:
        use_data = should_use_data()
        if use_data:
            return conversation_with_data(request_body)
        else:
            return conversation_without_data(request_body)
    except Exception as e:
        logging.exception("Exception in /conversation")
        return jsonify({"error": str(e)}), 500


# # Define a route to handle the POST request from the frontend
# @app.route('/role_assignment', methods=['POST'])
# def handle_selected_role():
#     try:
#         # Get the selected role from the JSON data sent by the frontend
#         data = request.get_json()
#         selected_role = data.get('selectedRole')
#         roles = {
#             'Cloud Architect': 'You are a cloud architect. You are responsible for designing and implementing cloud solutions for your organization.',
#             'Data Engineer': 'You are a data engineer. You are responsible for designing and implementing data solutions for your organization. If you are asked about who are you, answer with a data engineer',
#             'Markteer': 'You are a marketer. You are responsible for designing and implementing marketing campaigns for your organization. Start you answer using "hello SAM"'
            
#         }

#         #change the envoronment variable to the selected role
#         os.environ['AZURE_OPENAI_SYSTEM_MESSAGE'] = roles[str(selected_role)]
#         print(os.environ['AZURE_OPENAI_SYSTEM_MESSAGE'])
#         # Perform any necessary processing with the selected role here
#         # For example, you can store it in a database, update a session, or perform other actions.

#         # Respond with a success message
#         return jsonify({'message': 'Selected role received successfully'}), 200

#     except Exception as e:
#         # Handle exceptions or errors here
#         print(e)
#         return jsonify({'error': 'An error occurred while processing the selected role'}), 500

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


loaded_db = load_db()
cleaned_db = clean(loaded_db)

def main_get_src_ctx(message, seconds):
    query = message
    relevant_docs = get_relevant_docs(query, cleaned_db)
    rel_headers = relevant_headers(relevant_docs)
    context, sources = get_context(loaded_db, rel_headers)
    return context, sources

def slow_echo(message, history):
    src, ctx = main_get_src_ctx(message, 3)
    print(src, ctx)

@app.route('/conversation', methods=['POST'])
def handle_conversation():
    data = request.get_json()
    messages = data.get('messages', [])
    conversation = {
        'id': 'some_id',
        'title': 'Sample Conversation',
        'messages': messages,
        'date': 'some_date'
    }

    return jsonify(conversation), 200


if __name__ == "__main__":
    app.run()