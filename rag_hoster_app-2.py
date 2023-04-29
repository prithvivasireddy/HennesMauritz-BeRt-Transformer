#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## rag_hoster_app.py
import streamlit as st
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, RagTokenizer,
                          RagSequenceForGeneration, pipeline)

# Load the fine-tuned DistilBERT model and tokenizer
distilbert_model = AutoModelForQuestionAnswering.from_pretrained('/path/to/your/distilbert/output/dir')
distilbert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

# Create pipelines for both models
distilbert_qa_pipeline = pipeline('question-answering', model=distilbert_model, tokenizer=distilbert_tokenizer, device=0)

# Streamlit app
st.title('DistilBERT  Answering App')

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

