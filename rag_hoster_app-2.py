## distilbert_hoster_app.py
import streamlit as st
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, pipeline)

# Load the fine-tuned DistilBERT model and tokenizer
distilbert_model = AutoModelForQuestionAnswering.from_pretrained('/Users/prithvi/Desktop/HnM_BerT/Bert_Model')
distilbert_tokenizer = AutoTokenizer.from_pretrained('/Users/prithvi/Desktop/HnM_BerT/Bert_Tokenizer')

# Create pipeline
distilbert_qa_pipeline = pipeline('question-answering', model=distilbert_model, tokenizer=distilbert_tokenizer, device=0)

# Streamlit app
st.title('DistilBERT Answering App')

question = st.text_input('Enter your question:')

if st.button('Ask'):
    if question:
        answer = distilbert_qa_pipeline(question, context=None)
        st.write(f"Answer: {answer['answer']}")
    else:
        st.write('Please enter a question.')
