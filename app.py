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
An excerpt from a document is given below.

---------------------
{context}
---------------------

Given the document excerpt, answer the following query.
If the context does not provide enough information, decline to answer and direct the user to ask a Team Leader or a Manager.
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


# Function to build and cache the index from PDFs in a directory
@st.cache_resource
def build_and_cache_index():
    """Builds and caches the index from documents in the specified directory."""
    all_combined_texts = []
    data_folder = Path("data")

    # Ensure the 'Data' folder exists
    if not data_folder.exists() or not data_folder.is_dir():
        print(f"Error: The folder '{data_folder}' does not exist or is not a directory.")
        return None, None

    csv_files = list(data_folder.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{data_folder}'.")
        return None, None

    for csv_file_path in csv_files:
        print(f"Processing {csv_file_path.name}...")
        try:
            df = pd.read_csv(csv_file_path)

            # Assuming these columns *will* be present and correctly formatted
            df["combined"] = (
                "Title: " + df['Section/Column'].astype(str).str.strip() + " " +
                df['Name'].astype(str).str.strip() + "; Content: " +
                df['Notes'].astype(str).str.strip()
            )
            all_combined_texts.extend(df["combined"].tolist())

        except Exception as e:
            print(f"An error occurred while processing '{csv_file_path.name}': {e}")
            # Decide whether to skip this file or stop entirely
            continue # Skip to the next file if an error occurs

    if not all_combined_texts:
        print("No combined text could be extracted from any CSV file.")
        return None, None

    # Join all extracted combined texts into one large string for chunking
    full_text = "\n".join(all_combined_texts)

    chunk_size = 500
    chunks = [full_text[i : i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    embeddings = np.array([embed(chunk) for chunk in chunks])
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks


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

    query = st.chat_input("Ask something about your PDF")

    if not st.session_state.messages:
        add_message("Ask me anything!")

    if query:
        add_message(query, agent="human", stream=False, store=True)
        response = reply(query, index, chunks)
        add_message(response, agent="ai", stream=True, store=True)


if __name__ == "__main__":
    main()
