#!/usr/bin/env python
# coding: utf-8

## rag_hoster_app.py
import streamlit as st
import urllib.request
import os
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, pipeline)

# Load the BERT model and tokenizer from GitHub
bert_model_url = 'https://github.com/prithvivasireddy/hmrecc/raw/main/Bert_Model'
bert_tokenizer_url = 'https://github.com/prithvivasireddy/hmrecc/raw/main/Bert_Tokenizer'

bert_model_path = 'bert_model'
bert_tokenizer_path = 'bert_tokenizer'

if not os.path.exists(bert_model_path):
    os.makedirs(bert_model_path)
if not os.path.exists(bert_tokenizer_path):
    os.makedirs(bert_tokenizer_path)

urllib.request.urlretrieve(f'{bert_model_url}/config.json', f'{bert_model_path}/config.json')
urllib.request.urlretrieve(f'{bert_model_url}/pytorch_model.bin', f'{bert_model_path}/pytorch_model.bin')

urllib.request.urlretrieve(f'{bert_tokenizer_url}/config.json', f'{bert_tokenizer_path}/config.json')
urllib.request.urlretrieve(f'{bert_tokenizer_url}/vocab.txt', f'{bert_tokenizer_path}/vocab.txt')

bert_model = AutoModelForQuestionAnswering.from_pretrained(bert_model_path)
bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_path)

# Create pipelines for both models
distilbert_qa_pipeline = pipeline('question-answering', model=bert_model, tokenizer=bert_tokenizer, device=0)

# Streamlit app
st.title('DistilBERT Answering App')

question = st.text_input('Enter your question:')
model_choice = st.selectbox('Select model:', ('DistilBERT', 'RAG'))

if st.button('Ask'):
    if question:
        if model_choice == 'DistilBERT':
            answer = distilbert_qa_pipeline(question, context=None)
        elif model_choice == 'RAG':
            answer = rag_qa_pipeline(question, context=None)

        st.write(f"Answer: {answer['answer']}")
    else:
        st.write('Please enter a question.')
