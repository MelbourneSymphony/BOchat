import streamlit as st
from google import genai
import time
from google.api_core import exceptions
import os

# 1. Setup Gemini
gemini_token = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=gemini_token)

# 2. Load the entire manual once
@st.cache_data
def load_full_manual():
    # Assuming your manual is a text file or CSV
    with open("data/Box Office Manual (2025) - working.txt", "r") as f:
        return f.read()

manual_context = load_full_manual()

# 3. Simple Reply Function (No Chunks!)
def reply(query: str):
    # We put the manual directly into the system instructions
    sys_instruct = f"You are a helpful Box Office Assistant. Use this manual to answer: \n\n {manual_context}"
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=formatted_prompt
            )
            return response.text
        except exceptions.ServiceUnavailable:
            if attempt < 2:
                time.sleep(2)  # Wait 2 seconds before trying again
                continue
            else:
                return "The server is a bit busy right now. Please try your question again in a moment!"

# 4. Streamlit UI
query = st.chat_input("How can I assist you with box office operations today?")
if query:
    st.chat_message("user").write(query)
    answer = reply(query)
    st.chat_message("ai").write(answer)