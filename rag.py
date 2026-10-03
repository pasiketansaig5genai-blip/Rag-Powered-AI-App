import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

if not os.path.exists("./chroma_db"):
    from ingest import build_vector_db
    build_vector_db()


def process_uploaded_files(uploaded_files):
    """Processes uploaded files and returns a list of chunked Document objects."""
    docs = []
    
    for uploaded_file in uploaded_files:
        # Save temporary file locally to allow LangChain loaders to read it
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if suffix.lower() == ".pdf":
                loader = PyPDFLoader(tmp_path)
                docs.extend(loader.load())
            elif suffix.lower() in [".txt", ".md"]:
                loader = TextLoader(tmp_path)
                docs.extend(loader.load())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Chunk the uploaded document content
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks


def build_dynamic_rag_chain(chunks):
    """Creates an in-memory or persisted vector store from user chunks and builds the RAG chain."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Build Chroma store with uploaded chunks
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # System prompt enforcing strict grounding on uploaded docs
    template = """You are a helpful assistant grounding your answers strictly in the context provided below.
If the context does not contain the answer, state clearly that you cannot find the answer in the uploaded document.

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def get_rag_chain():
    # 1. Initialize Embeddings using the working model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 2. Load existing Chroma DB
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Setup LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 4. Prompt Template
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # Helper function to format retrieved docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Build LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# System prompt engineered for strict grounding (AI-2)
SYSTEM_PROMPT = """You are a specialized grounded AI assistant.

Context:
{context}

Instructions:
1. Answer the user's question relying strictly on the context provided above.
2. If the answer cannot be found in the context, explicitly state: "I cannot find information about this in the provided knowledge base."
3. Do not make up facts or use general world knowledge.

Question: {question}
"""

def query_rag_system(question: str):
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY") )

    
    # Connect to existing Chroma database
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Retrieve context documents
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # LLM Initialization (AI-1 API Integration)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)

    return response.content, docs