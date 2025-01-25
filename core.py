import os
import pickle
import subprocess
import sys
import streamlit as st
import nltk
import time

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_huggingface import HuggingFaceEmbeddings

# API key input
GOOGLE_API_KEY = st.sidebar.text_input("Enter Google API Key:", type="password")

# Initialize Streamlit app
st.title("_READ LESS_")
st.caption("Chat, search, and save time—no more endless reading!")
st.sidebar.title("Input URLs")

# Set up session state for chat
if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

# URL inputs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vector_index.pkl"

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.9,
    max_tokens=500,
    max_retries=2,
)
status_container = st.container()

# Process URLs
if process_url_clicked:
    if any(urls):
        try:
            with status_container.status("Processing URLs...") as status:
                st.sidebar.info("Processing URLs...")

                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()
                st.write("Data Loaded.")
                status.update(expanded=True)
                
                # Split data into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(data)
                st.write("Data Split into Chunks.")

                # Create embeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.sidebar.info("Embeddings created. Checking dimensionality...")
                
                # Check embedding dimensionality
                embedding_example = embeddings.embed_query("This is a test sentence.")
                embedding_dim = len(embedding_example)  # Get embedding vector size
                #st.sidebar.info(f"Embedding Dimensionality: {embedding_dim}")

                # Ensure dimensionality is correct
                if embedding_dim <= 0:
                    raise ValueError("Embedding dimensionality is invalid!")

                # Create FAISS index
                vector_index = FAISS.from_documents(docs, embeddings)
                st.write("Embeddings and FAISS index created successfully.")
                time.sleep(1)
                status.update(label="Processing complete!", state="complete", expanded=False)
                
                # Save vector index locally
                with open(file_path, "wb") as f:
                    pickle.dump(vector_index, f)
                st.sidebar.success("URLs processed successfully!")

        except Exception as e:
            st.sidebar.error(f"Error while processing URLs: {e}")
    else:
        st.sidebar.error("Please enter valid URLs.")

# Chat UI
chat_placeholder = st.empty()

with chat_placeholder.container():
    for i in range(len(st.session_state["past"])):  
        message(st.session_state["past"][i], is_user=True, key=f"user_{i}")
        message(st.session_state["generated"][i], key=f"bot_{i}")

# Input for user query
user_query = st.chat_input("Ask your question:", key="user_query")

if user_query:
    # Show user query instantly
    st.session_state["past"].append(user_query)
    message(user_query, is_user=True, key=f"user_{len(st.session_state['past']) - 1}")

    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vector_index = pickle.load(f)

            st.sidebar.info("Starting query processing...")

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
            
            # Process the query and generate the response
            result = chain.invoke({"question": user_query}, return_only_outputs=True)
            answer = result["answer"]
            sources = result.get("sources", "No sources available.")
            
            # Store the answer after it's generated
            st.session_state["generated"].append(f"{answer}\n\nSources:\n{sources}")

            # Display the bot's answer
            message(f"{answer}\n\nSources:\n{sources}")

        except Exception as e:
            st.sidebar.error(f"Error while processing the query: {e}")
    else:
        st.sidebar.error("Vector index file not found. Please process URLs first.")