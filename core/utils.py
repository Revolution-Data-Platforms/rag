import os, re, requests
from tqdm import tqdm
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings



def decode_base64(encoded_link):
    cmd  = f'echo {encoded_link} | base64 --decode'
    stream = os.popen(cmd)
    output = stream.read()
    return output

def query_AzCogSearch(query, k=4):
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import QueryType
    service_name = "rdpdemo01"
    query_key = "ho4AX2ixS4Bn3QyJYoyWa1ordS6ddd5MIplV2pujbyAzSeDHaEro"

    index_name = "adlsgen2-index"

    # Create an SDK client
    endpoint = "https://{}.search.windows.net/".format(service_name)

    search_client = SearchClient(endpoint=endpoint,
                      index_name=index_name,
                      credential=AzureKeyCredential(query_key))
    
    # Search documents
    results = search_client.search(search_text=query, 
                               top=k, 
                               query_type=QueryType.SEMANTIC, 
                               query_language="en-us",
                               query_speller="lexicon",
                               semantic_configuration_name="semantic config")
    return results

def AzCogSeQuery(query, k=4):
    service_name = "rdpdemo01"
    query_key = "ho4AX2ixS4Bn3QyJYoyWa1ordS6ddd5MIplV2pujbyAzSeDHaEro"

    index_name = "adlsgen2-index"
    BASE_PATH = '/Ai/MSFT_demo/markdowns/azure-docs/articles'

    url = f"https://{service_name}.search.windows.net/indexes/{index_name}/docs/search?api-version=2020-06-30-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": query_key
    }
    body = {
        "search": query,
        "queryType": "semantic",
        "searchFields": "content",
        "queryLanguage": "en-us",
        'top': k
    }
    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 200:
        k_response = response.json()['value']
        text = []
        sources = []
    for each_response in k_response:
        Doc_RELATIVE_PATH = decode_base64(each_response['metadata_storage_path']).split('articles/')[1].split('.')[0]
        
        for caption in each_response['@search.captions']:
            caption = caption['text']
            sources.append((Doc_RELATIVE_PATH, caption))
            text.append(GetParagraph(caption, f'{BASE_PATH}/{Doc_RELATIVE_PATH}.md'))
    return text, sources

def GetParagraph(caption, md_path):
    try:
        with open(md_path, 'r') as f:
            contents = f.read()
            # split the contents into front matter and Markdown content
            front_matter, md_content = contents.split('---\n', 2)[1:]     

        # split the caption into sentences
        words = caption.split(' ')
        
        # get the first and last 3 words
        first_words = ' '.join(words[:3])
        last_words = ' '.join(words[-3:])
        stopping_char = "## "

        len_first_words = len(first_words)
        len_last_words = len(last_words)

        pattern = re.escape(stopping_char) + r'(.*?)' + re.escape(first_words)
        match = re.search(pattern, md_content, re.DOTALL)
        if match:
            from_last_header_text = match.group(0)
            len_from_last_header_text = len(from_last_header_text)
        else:
            from_last_header_text = ''
            len_from_last_header_text = 0

        # Selecting the rest of the text till the next heading
        pattern = re.escape(last_words) + r'(.*?)' + re.escape(stopping_char)
        match = re.search(pattern, md_content, re.DOTALL)
        if match:
            to_next_header_text = match.group(0)
        else:
            to_next_header_text = ''
        try:
            return from_last_header_text[:(len_from_last_header_text - len_first_words)] + caption + to_next_header_text[len_last_words:]
        except:
            return from_last_header_text + caption + to_next_header_text
    except:
        return caption
    
def clean_text(text):
    text = re.sub(r'#', '', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\t', '', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def get_answer(endpoint, question):
    headers = {
    'Content-Type': 'application/json',
    }
    data = {'question': question}
    response = requests.post(endpoint, params=data, headers=headers)
    return response.json()


def Az_prompt(question):
    # Querying Azure Cognitive Search
    AzCogSrch_reponse, sources = AzCogSeQuery(question, k=2)   

    # Concatenating the responses
    context_data_string = '\n\n\n'.join([reponse for reponse in AzCogSrch_reponse])      

    # Chunking the responses into 1000 character chunks & Saving the chunks as documents
    markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=300)
    docs = markdown_splitter.create_documents([context_data_string], 
                                            metadatas=[{'source': 'Azure Cognitive Search'} for i in range(len(context_data_string))])

    doc_search = Chroma.from_documents(docs, HuggingFaceEmbeddings())

    # Getting K=7 most similar documents
    if len(docs) > 7:
        doc_Chroma = doc_search.similarity_search(question, k=7)
    else:
        doc_Chroma = doc_search.similarity_search(question, k=len(docs))

    # Concatenating all selected documents
    doc_Chroma_string = '\n'.join([doc.page_content for doc in doc_Chroma])

    ############################

    # Prompting the model with the question and the selected documents
    prompt = """Given the following extracted parts from a long document and a question. Create a final answer for the question. if the answer is not found in the extracted parts, then answer with "I do not know. There is not enough context".

question: {question}

CONTEXT: {context_data} 

FINAL ANSWER:
""".format(question = question,
                  context_data=clean_text(doc_Chroma_string))
    
    return prompt, sources

def search_result(i: int, 
                  url: str, 
                  title: str, 
                  highlights: str,
                  author: str, 
                  length: str, **kwargs) -> str:
    """ HTML scripts to display search results. """
    return f"""
        <div style="font-size:120%;">
            {i + 1}.
            <a href="{url}">
                {title}
            </a>
        </div>
        <div style="font-size:95%;">
            <div style="color:grey;font-size:95%;">
                {url[:90] + '...' if len(url) > 100 else url}
            </div>
            <div style="float:left;font-style:italic;">
                {author} ·&nbsp;
            </div>
            <div style="color:grey;float:left;">
                {length} ...
            </div>
            {highlights}
        </div>
    """

def get_title(url: str) -> str:
    """ Return title of a web page. """
    import requests
    from bs4 import BeautifulSoup
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup.title.string
    except:
        return url
