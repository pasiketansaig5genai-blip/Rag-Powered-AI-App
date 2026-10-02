import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def build_vector_db():
    print("📂 Loading documents from data/ directory...")
    loader = DirectoryLoader('./data', glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("⚠️ No documents found in data/ folder!")
        return

    print("✂️ Chunking text documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print("🧠 Embedding and storing in Chroma DB...")
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"))

    
    # Store directly into chroma_db directory
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("✅ Vector DB build complete! Saved to ./chroma_db")

if __name__ == "__main__":
    build_vector_db()