import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import faiss
import csv
from huggingface_hub import InferenceClient
from faiss import IndexFlatL2
from mistralai.client import MistralClient
from mistralai import Mistral


hf_token = st.secrets["HF_TOKEN"]
ms_token = st.secrets["MS_TOKEN"]
client = Mistral(api_key=ms_token)
prompt = """
You are a helpful assistant that answers questions about from symphony orchestra box office attandants.
You are designed to assist with queries related to the box office, ticketing, and customer service information and policies.
An excerpt from the the box office manual is given below.


---------------------
{context}
---------------------

Given the document excerpt, answer the following query.
If the context does not provide enough information, decline to answer and direct the user to ask a Team Leader or a Supervisor.
Do not output anything that can't be answered from the context.

Query: {query}
Answer:
"""

# Initialize session state variables if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to add a message to the chat
def add_message(msg, agent="ai", stream=True, store=True):
    """Adds a message to the chat interface, optionally streaming the output."""
    if stream and isinstance(msg, str):
        msg = stream_str(msg)

    with st.chat_message(agent):
        if stream:
            output = st.write_stream(msg)
        else:
            output = msg
            st.write(msg)

    if store:
        st.session_state.messages.append(dict(agent=agent, content=output))


# Function to stream a string with a delay
def stream_str(s, speed=250):
    """Yields characters from a string with a delay to simulate streaming."""
    for c in s:
        yield c
        time.sleep(1 / speed)


# Function to stream the response from the AI
def stream_response(response):
    """Yields responses from the AI, replacing placeholders as needed."""
    for r in response:
        content = r.choices[0].delta.content
        # prevent $ from rendering as LaTeX
        content = content.replace("$", "\$")
        yield content


# Decorator to cache the embedding computation
@st.cache_data
def embed(text):
    client = InferenceClient(model="sentence-transformers/all-MiniLM-L6-v2", token=hf_token)
    output = client.feature_extraction(text)
    return output


# Function to call embeddings and build the index
@st.cache_resource
def build_and_cache_index():
    """Loads the index and chunks from the embeddings folder. If not found, throws an error and does not build."""
    import numpy as np
    from pathlib import Path

    embeddings_folder = Path("embeddings")
    chunks_path = embeddings_folder / "chunks.npy"
    embeddings_path = embeddings_folder / "embeddings.npy"

    # Try to load embeddings and chunks from disk
    if embeddings_folder.exists() and chunks_path.exists() and embeddings_path.exists():
        chunks = np.load(chunks_path, allow_pickle=True).tolist()
        embeddings = np.load(embeddings_path)
        dimension = embeddings.shape[1]
        index = IndexFlatL2(dimension)
        index.add(embeddings)
        return index, chunks
    else:
        print("Error: Embeddings not found in 'embeddings' folder. Please generate them before running the app.")
        return None, None


# Function to reply to queries using the built index
def reply(query: str, index: IndexFlatL2, chunks):
    embedding = embed(query)
    embedding = np.array([embedding])

    _, indices = index.search(embedding, k=2)
    context = [chunks[i] for i in indices[0]]

    user_message = prompt.format(context=context, query=query)

    messages = [{"role":"user", "content":user_message}]
    chat_response = client.chat.complete(model="mistral-medium", messages=messages)
    return chat_response.choices[0].message.content


# Main application logic
def main():
    """Main function to run the application logic."""
    if st.sidebar.button("🔴 Reset conversation"):
        st.session_state.messages = []

    index, chunks = build_and_cache_index()

    if index is None or chunks is None:
        st.error("Failed to build the document index. Please check that the 'data' folder exists and contains valid CSV files.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["agent"]):
            st.write(message["content"])

    query = st.chat_input("Ask me anything.")

    if not st.session_state.messages:
        add_message("How can I assist you with box office operations today?")

    if query:
        add_message(query, agent="human", stream=False, store=True)
        response = reply(query, index, chunks)
        add_message(response, agent="ai", stream=True, store=True)


if __name__ == "__main__":
    main()
