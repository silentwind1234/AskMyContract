import streamlit as st
import tiktoken
import numpy as np
import openai
from dotenv import load_dotenv
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
from io import StringIO 
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from pdfManager import pdfFile
from promptManager import Prompt
import pandas as pd
import json
# Load environment variables
load_dotenv()

# Configure Azure OpenAI Service API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# configure UI elements with Streamlit
st.markdown(""" <style> div.stButton > button:first-child { background-color: #008CBA; color: white; position: absolute; width: 60%; left: 20%; } </style>""", unsafe_allow_html=True) 
st.markdown("<h1 style='text-align: center; color: black;'>Offers Summary</h1>", unsafe_allow_html=True)
files = st.file_uploader("Please choose a file",accept_multiple_files=True)
st.write(" ")

with st.container():
    askBtn = st.button('Summarize',key='but_a')
st.write(" ")
st.write(" ")
answerArea = st.empty()
answerArea.info("")

pdfFileValid = True
pdfs = []

ready = False
if files is not None:
    pdfs = []
    print(len(files))
    counter = 0
    answerArea.info(f'''Loading files {len(files)} Please Wait...''')
    
    for file in files:
        counter += 1
        answerArea.empty()
        answerArea.info(f'''Loading {counter} of {len(files)} Please Wait...''')
        pdf = pdfFile(file)
        pdfFileValid = pdf.check()
        if not pdfFileValid:
            answerArea.empty()
            answerArea.info("Invalid PDF file, please only use PDF files")
        else:
            if not ready:
                pdf.loadPages()
                pdfs.append(pdf)
    answerArea.empty()
    answerArea.info("Files Ready. Ready to Generate Summary!")
    ready = True

if askBtn and pdfFileValid and ready:
    answerArea.empty()
    answerArea.info(f'''Generating summary for {len(pdfs)} Please Wait...''')
    counter = 0
    answers = []
    for pdf in pdfs:
        counter += 1
        answerArea.empty()
        answerArea.info(f'''Generating summary for {counter} of {len(pdfs)} Please Wait...''')
        page = pdf.fullText
        prompt = f"""
        Offer Pages:
        {page}
        
        The above offer pages is part of a commercial offer sent to Siemens energy.

        give the answer in the following format only, do not give any other text or code in your answer other than the requested data:
            
            Supplier Name # Offer Value in number only # Currency in format of XXX # Offer validity in date format of dd/MMM/yyyy # delivery period in XX months # warrenty period in XX months # summerized scope in 1 sentence # summerized deviations and special conditions in 20 words maximum
        """
        promptObj = Prompt()
        offers = []
        answer = promptObj.run_prompt(prompt,max_tokens=250)
        answer = answer.replace("\n","")
        answer = answer.replace("Answer:","")
        
        arr = answer.split("#")
        answers.append(arr)
        print(arr)
    np_array = np.array(answers)
    df = pd.DataFrame(np_array,columns=["Supplier","Value","Currency","Validity","Delivery Period","Warrenty","Scope","Special conditions"])
    answerArea.empty()
    answerArea.info("Result Exported to CSV file")
    df.to_csv('offers1.csv', encoding='utf-8', index=False)
else:
    if not ready:
        answerArea.empty()
        answerArea.info("Please Choose a file then press Summarize")