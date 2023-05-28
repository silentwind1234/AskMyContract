import tiktoken
import openai
from dotenv import load_dotenv
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
from io import StringIO 
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from pdfManager import pdfFile

class Prompt:
    def __init__(self) -> None:
        self.completion_model = 'text-davinci-003'
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
    def run_prompt(self,prompt, max_tokens=1000):
        response = openai.Completion.create(
            engine=self.completion_model,
            prompt=prompt,
            temperature=0.2,
            max_tokens=max_tokens
        )
        return response['choices'][0]['text']

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
    def summerizeAnswer(self,prompt, max_tokens=1000):
        response = openai.Completion.create(
            engine=self.completion_model,
            prompt=prompt,
            temperature=0.2,
            max_tokens=max_tokens
        )
        return response['choices'][0]['text']