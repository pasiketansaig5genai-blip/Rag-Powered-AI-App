import os
import streamlit as st
from dotenv import load_dotenv
from rag import process_uploaded_files, build_dynamic_rag_chain

load_dotenv()

st.set_page_config(page_title="RAG-Powered AI App", page_icon="📄")
st.title("📄 Dynamic Document RAG Assistant")

# Sidebar for file uploads
with st.sidebar:
    st.header("📤 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files to query",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if st.button("Process & Index Documents"):
        if uploaded_files:
            with st.spinner("Processing documents and building vector store..."):
                try:
                    chunks = process_uploaded_files(uploaded_files)
                    st.session_state.rag_chain = build_dynamic_rag_chain(chunks)
                    st.session_state.messages = []  # Reset chat history for new docs
                    st.success(f"Successfully processed {len(uploaded_files)} file(s)!")
                except Exception as e:
                    st.error(f"Error processing files: {e}")
        else:
            st.warning("Please upload at least one document.")

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt execution
if prompt := st.chat_input("Ask a question about your uploaded document..."):
    if "rag_chain" not in st.session_state:
        st.error("Please upload and index a document in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")