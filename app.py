import os
import requests
import streamlit as st
import chromadb

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# BASIC SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="College Management Chatbot",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 College Management Chatbot")
st.write("Ask questions about college information.")


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# --------------------------------------------------
# CREATE CHROMA DATABASE
# --------------------------------------------------

@st.cache_resource
def get_collection():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="college_documents"
    )

    return collection


collection = get_collection()


# --------------------------------------------------
# READ PDF
# --------------------------------------------------

def read_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# SPLIT TEXT INTO CHUNKS
# --------------------------------------------------

def split_text(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# --------------------------------------------------
# ADD DOCUMENTS TO CHROMA
# --------------------------------------------------

def add_documents():

    folder = "documents"

    if not os.path.exists(folder):
        return 0

    pdf_files = [
        file for file in os.listdir(folder)
        if file.lower().endswith(".pdf")
    ]

    total_chunks = 0

    for file_name in pdf_files:

        file_path = os.path.join(folder, file_name)

        text = read_pdf(file_path)

        if not text.strip():
            continue

        chunks = split_text(text)

        for index, chunk in enumerate(chunks):

            document_id = f"{file_name}_{index}"

            # Check whether document already exists
            existing = collection.get(
                ids=[document_id]
            )

            if existing["ids"]:
                continue

            embedding = embedding_model.encode(
                chunk
            ).tolist()

            collection.add(
                ids=[document_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[
                    {
                        "source": file_name
                    }
                ]
            )

            total_chunks += 1

    return total_chunks


# --------------------------------------------------
# SEARCH RELEVANT DOCUMENTS
# --------------------------------------------------

def search_documents(question, number_of_results=3):

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results.get("documents", [[]])[0]

    return documents


# --------------------------------------------------
# ASK OLLAMA
# --------------------------------------------------

def ask_ollama(question, context):

    prompt = f"""
You are a helpful college management chatbot.

Answer the student's question using the information
provided in the context.

If the answer is not available in the context,
say:

"I could not find this information in the college documents."

Do not make up information.

Context:
{context}

Student Question:
{question}

Answer:
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "response",
            "Sorry, I could not generate an answer."
        )

    except requests.exceptions.ConnectionError:

        return (
            "Ollama is not running. "
            "Please start Ollama and try again."
        )

    except Exception as e:

        return f"Error while contacting Ollama: {e}"


# --------------------------------------------------
# ADD COLLEGE DOCUMENTS
# --------------------------------------------------

if st.sidebar.button("📚 Load College Documents"):

    with st.spinner("Reading college documents..."):

        count = add_documents()

    st.sidebar.success(
        f"Added {count} new document chunks."
    )


# --------------------------------------------------
# SHOW DATABASE INFORMATION
# --------------------------------------------------

st.sidebar.subheader("System Information")

st.sidebar.write(
    f"LLM Model: `{OLLAMA_MODEL}`"
)

st.sidebar.write(
    "Embedding: `all-MiniLM-L6-v2`"
)

st.sidebar.write(
    "Database: `ChromaDB`"
)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# --------------------------------------------------
# USER QUESTION
# --------------------------------------------------

question = st.chat_input(
    "Ask something about the college..."
)


if question:

    # Display user question

    with st.chat_message("user"):

        st.write(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Search documents

    relevant_documents = search_documents(
        question
    )


    # If no documents exist

    if not relevant_documents:

        answer = (
            "Please load the college documents first "
            "using the sidebar button."
        )

    else:

        context = "\n\n".join(
            relevant_documents
        )

        # Generate answer

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = ask_ollama(
                    question,
                    context
                )

            st.write(answer)


    # Save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )