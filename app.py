from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ------------------ LOAD PDF ------------------
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")
docs = loader.load()


# ------------------ SPLIT TEXT ------------------
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)


# ------------------ EMBEDDINGS (FREE - HUGGINGFACE) ------------------
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()


# ------------------ VECTOR STORE ------------------
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(chunks, embeddings)


# ------------------ RETRIEVER ------------------
# retriever = db.as_retriever()
retriever = db.as_retriever(search_kwargs={"k": 3}) #👉 Gets top 3 chunks → more accurate answers

# ------------------ LLM (OPENROUTER FREE MODEL) ------------------
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",   # ✅ your selected free model
    base_url="https://openrouter.ai/api/v1"
)


# ------------------ QA CHAIN ------------------
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

template = """Answer the question based only on the following context:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ------------------ ASK QUESTION ------------------
# query = "What is this document about?"

# result = qa.invoke(query)

# print("\nAnswer:\n", result)


# ---------------- UI ----------------


st.title("PDF Chatbot")

query = st.text_input("Ask a question")

if query:
    response = qa.invoke(query)
    st.write(response)