
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import os
import json
import tiktoken
import openai
import numpy as np
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt
import streamlit as st
# Load environment variables

# Define embedding model and encoding
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CHUNK_SIZE = 8000
COMPLETION_MODEL = 'text-davinci-003'

# initialize tiktoken for encoding text
encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)

class pdfFile:
    def __init__(self,file:str) -> None:
        self.file = file
        self.pages = []
        self.embeddings = []
    def check(self):
        try:
            PdfReader(self.file)
            return True
        except PdfReadError:
            return False
    def loadPages(self):
        reader = PdfReader(self.file)
        for page in reader.pages:
            content = page.extract_text()
            content = content.replace("\n", " ")
            content = content.replace("  ", " ")
            self.pages.append(content)
    
    def getCost(self):
        totalTokens = 0
        for doc in self.pages:
            num_tokens = len(encoding.encode(doc))
            totalTokens += num_tokens
        return totalTokens
    
    def createEmbeddings(self):
        self.embeddings = []
        for page in self.pages:
            response = openai.Embedding.create(input=page, engine='text-embedding-ada-002')    
            self.embeddings.append(response['data'][0]['embedding'])

    def getSimilarities(self,text):
        newtext = text.replace("\n", " ")
        newtext = newtext.replace("  ", " ")
        ques = openai.Embedding.create(input=newtext, engine='text-embedding-ada-002')['data'][0]['embedding']
        similarities = []
        for embedding in self.embeddings:
            similarities.append(cosine_similarity(ques,embedding))
        return similarities
    
    def getMaxSimilarity(self,text):
        if len(self.embeddings) == 0:
            return ""
        max_i = np.argmax(self.getSimilarities(text=text))
        return self.pages[max_i]