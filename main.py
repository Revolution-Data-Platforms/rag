
from backend.retrieval.ciena_retreival import CienaRetrieval
from backend.embedder.baseEmbedder import baseEmbedder
from backend.retrieval.utils import *
from backend.retrieval.rereanker import Reranker
from langchain.document_loaders import JSONLoader

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
            if "10Aug-BP_Engineering_guide" in file or "Blue_Planet_MLA_Cloud_Deployment_Guide" in file:
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
                    except:
                        print(f"error in loading  file {file_name}")
                        pass

    return loaded_data


def clean(docs):
    loaded_data = filter_empty(docs)
    loaded_data = filter_redundant(loaded_data)
    loaded_data = exclude_toc(loaded_data)
    return loaded_data


def get_relevant_docs(query, docs):
    """Get relevant documents from Ciena database."""
    ciena_retrieval = CienaRetrieval(**retriael_kwargs)
    relevant_docs = ciena_retrieval.get_res(query, docs)
    reranked_res = reranker.rerank(query, relevant_docs)
    return reranked_res

def get_context(docs):
    context = ciena_retreival.get_context(docs)
    return context




if __name__ == "__main__":
    loaded_db = load_db()
    # cleaned_db = clean(loaded_db)
    query = "Table BPO Runtime License"
    relevant_docs = get_relevant_docs(query, loaded_db)
    context = get_context(relevant_docs)
    print(context)
