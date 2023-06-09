
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
import fitz
import sys
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
        self.fullText = ""
        self.sentences = []
        self.pages = []
        self.blocks = []
        self.embeddings = []
        self.similarities = []
        
    def check(self):
        try:
            PdfReader(self.file)
            return True
        except PdfReadError:
            return False
    def loadPages(self):
        reader = PdfReader(self.file)
        self.sentences = []
        self.pages = []
        self.fullText = ""
        
        for page in reader.pages:
            content = page.extract_text()
            content = content.replace("\n", " ")
            content = content.replace("  ", " ")
            splited = content.split(".")

            self.sentences += [item for item in splited if len(item) > 10]
            
            self.pages.append(content)
        self.fullText = " ".join(self.pages)
        return self.pages
    
    def getCost(self):
        totalTokens = 0
        for doc in self.pages:
            num_tokens = len(encoding.encode(doc))
            totalTokens += num_tokens
        return totalTokens*0.0004/1000
    
    def createEmbeddings(self):
        
        filename = f'''{self.file.name}.npy'''
        if len(self.pages) > 0:
            try:
                stored_embeddings = np.load(filename)
                self.embeddings = stored_embeddings
            except:
                self.embeddings = []
                for page in self.pages:
                    response = openai.Embedding.create(input=page, engine='text-embedding-ada-002')    
                    self.embeddings.append(response['data'][0]['embedding'])
                all_embeddings = np.array(self.embeddings)
                np.save(filename, all_embeddings)

    def getSimilarities(self,text):
        newtext = text.replace("\n", " ")
        newtext = newtext.replace("  ", " ")
        ques = openai.Embedding.create(input=newtext, engine='text-embedding-ada-002')['data'][0]['embedding']
        self.similarities = []
        for embedding in self.embeddings:
            self.similarities.append(cosine_similarity(ques,embedding))
        
        return self.similarities
    
    def getMaxSimilarity(self,text):
        if len(self.embeddings) == 0:
            return ""
        max_i = np.argmax(self.getSimilarities(text=text))
        
        if self.similarities[max_i] < 0.6:
            return ""
        else:
            return self.pages[max_i]
        
    def getMaxSimilaritySentences(self,text,top:int):
        print(self.embeddings)
        if len(self.embeddings) == 0:
            return ""
        searchStr = []
        list = self.getSimilarities(text=text)
        
        sorted_values = np.argsort(-list)
        for x in range(top):
            if x < len(sorted_values):
                searchStr.append(self.blocks[sorted_values[x]])
        st.code("\n".join(searchStr))
        if len(searchStr) > 0:
            return ""
        else:
            return "\n\n".join(searchStr)