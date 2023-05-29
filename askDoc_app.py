import streamlit as st
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
# Load environment variables
load_dotenv()

# Configure Azure OpenAI Service API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# configure UI elements with Streamlit
st.markdown(""" <style> div.stButton > button:first-child { background-color: #008CBA; color: white; position: absolute; width: 60%; left: 20%; } </style>""", unsafe_allow_html=True) 
st.markdown("<h1 style='text-align: center; color: black;'>Ask My Contract</h1>", unsafe_allow_html=True)
file = st.file_uploader("Please choose a file")
st.write(" ")

with st.container():
    questionArea = st.text_area('My Request', height=300, max_chars=4000,placeholder="My legal Baseline",label_visibility='hidden')
    askBtn = st.button('Ask',key='but_a')
st.write(" ")
st.write(" ")
answerArea = st.empty()
answerArea.info("")

pdfFileValid = True
pdf = pdfFile('')
ready = False
if file is not None:
    pdf = pdfFile(file)
    pdfFileValid = pdf.check()
    if not pdfFileValid:
        answerArea.empty()
        answerArea.info("Invalid PDF file, please only use PDF files")
    else:
        if not ready:
            answerArea.empty()
            answerArea.info("Please Wait...")
            pdf.loadPages()
            # st.code(f'''Total Cost ${pdf.getCost()}''')
            pdf.createEmbeddings()
            answerArea.empty()
            answerArea.info("File Ready. Please ask you question!")
            ready = True
        
    
if askBtn and pdfFileValid and len(questionArea) > 0 and ready:
    answerArea.empty()
    answerArea.info("Please Wait...")
    page = pdf.getMaxSimilarity(text=questionArea)
    answerArea.empty()
    answerArea.info("Similar Page Found, Please Wait...")
    prompt = f"""
    Contract Page:
    {page}
    
    The above contract page is part of a contract between Siemens energy (as the contractor) and the Owner.

    {questionArea}
    Answer:"""
    promptObj = Prompt()

    answer = promptObj.run_prompt(prompt)
    
    answerArea.empty()
    answerArea.info(answer)
    
else:
    if not ready:
        answerArea.empty()
        answerArea.info("Please Choose a file then ask your question")