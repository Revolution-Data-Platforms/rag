import streamlit as st

# Langchains
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings

# RDP Specific
from core.DataMind import DataMind
from core.LLMS import Remote_LLM
from core.templates import (load_css, search_result)


def streamlit_UI(embedding_model, vectorstore_location, LLM_endpoint, LLM_kwargs={}, embedding_kwargs={"device": "cuda"}):
    
    
    @st.cache_resource
    def load_embeddings(embedding_model, model_kwargs={"device": "cuda"}):
        return HuggingFaceInstructEmbeddings(model_name=embedding_model,
                                                        model_kwargs=model_kwargs)
    @st.cache_resource
    def load_FAISS_vectorstore(vectorstore_location, _emdeddings):
        return FAISS.load_local(vectorstore_location,
                                    _emdeddings)
    @st.cache_resource
    def init_LLM(llm_endpoint, LLM_kwargs):
        return Remote_LLM(endpoint=llm_endpoint, generation_config=LLM_kwargs)
    
    embeddings = load_embeddings(embedding_model, embedding_kwargs)
    vectorstore = load_FAISS_vectorstore(vectorstore_location, embeddings)
    LLM_instance = init_LLM(LLM_endpoint, LLM_kwargs)

    return embeddings, vectorstore, LLM_instance



def streamlit_QA(context_generator, LLM_instance, system_prompt, assistant_token):


    # UI stuff 
    ##########################################################
    st.write(load_css(), unsafe_allow_html=True)
    
    # Logo and Title
    col1, mid, col2 = st.columns([1,1,18])
    with col1:
        st.image('../../UI/assets/rpd_logo.png', width=70)
    with col2:
        st.markdown('## DataMind', unsafe_allow_html=True)

    q_search = st.text_input('Enter your question here:')
    ##########################################################

    if len(q_search)>0:
        if st.button("Answer",type='primary'):

            # Prepare Prompt & Sources
            PROMPT, sources = context_generator.generate_prompt_with_context_and_sources(
                system_prompt=system_prompt,
                question=q_search,
                assistant_token=assistant_token
            )

            print("Prompt >>>\n\n", PROMPT)

            with st.spinner(text="This may take a moment ..."):
                # Get Answers
                VICUNA = LLM_instance(PROMPT)
            print("Answer >>>\n\n", VICUNA)
            # Getting only the Answer
            VICUNA = VICUNA.split(assistant_token)[1]
            VICUNA = VICUNA.replace('</s>', '')
            

            st.write("## RDP-FineTuned-Falcon-7B:")
            if 'I do not know' in VICUNA \
                or "not mentioned in the provided context" in VICUNA \
                or "not provide enough context" in VICUNA:
                st.warning(f'\n It seems that the question you asked is out of the document context! Please try to ask the question in a different way.', icon="❌")
            else:
                st.success(f'\n{VICUNA}', icon="✅")
                st.write('## Sources:')
                st.write('---')
                printed_sources = []
                for idx, source in enumerate(sources):
                    source, page = source
                    source = source.replace('\x00', '')
                    splitted_url = source.split(".")
                    # remove "x00" from the splitted_url
                    # splitted_url = [url.replace('\x00', '') for url in splitted_url]
                    ext = splitted_url[-1].replace('\x00', '')
                    link = ".".join(splitted_url[:-1])


                    if 'pdf' in ext:
                        url = link + '.pdf'
                    elif 'docx' in ext:
                        url = link + '.docx'
                    elif 'doc' in ext:
                        url = link + '.doc'
                    else:
                        url = link + ext
                    if url not in printed_sources:
                        st.write(search_result(idx, url, f"Document: {url.split('/')[-1].replace('%20', ' ')} - Page: {page}", '')
                                                    , unsafe_allow_html=True)
                        st.write('---')
                        printed_sources.append(url)

    
if __name__ == '__main__':
    
    strm = streamlit_UI(embedding_model='hkunlp/instructor-xl',
                        vectorstore_location='/DataMind/Customers/to_ciena/FAISS/DocStore',
                        LLM_endpoint='http://mavrik.specgood.ai:8000/answer')
    
    context_generator = DataMind(vectore_store=strm.load_FAISS_vectorstore(strm.load_embeddings()), 
                                 embedding=strm.load_embeddings(),
                                 k=3)
    
    strm.streamlit_QA(context_generator)

    
