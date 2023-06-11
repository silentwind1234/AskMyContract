import streamlit as st
from streamlit_chat import message
import tiktoken
import openai
from dotenv import load_dotenv
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
from io import StringIO 
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from pdfManager import pdfFile
from promptManager import Prompt
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
# Load environment variables
load_dotenv()

# Configure Azure OpenAI Service API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

st.markdown(""" <style> div.stButton > button:first-child { background-color: #008CBA; color: white; position: absolute; width: 60%; left: 20%; } </style>""", unsafe_allow_html=True) 
st.markdown("<h1 style='text-align: center; color: black;'>Chat with My Documents</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Please choose a file", type="pdf")
st.write(" ")

if uploaded_file:
    # pdf = pdfFile(uploaded_file)
    data = PyPDFLoader(uploaded_file.name).load_and_split()
    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=os.getenv("OPENAI_API_KEY")),
                                                                      retriever=vectors.as_retriever())

    def conversational_chat(query):
        
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk about your data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")