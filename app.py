import os
import streamlit as st
from dotenv import load_dotenv

# Import your QA chain function from rag.py
from rag import get_rag_chain

load_dotenv()

st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖")
st.title("💬 RAG AI Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load RAG Chain
@st.cache_resource
def load_chain():
    return get_rag_chain()

rag_chain = load_chain()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input
if prompt := st.chat_input("Ask something about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})