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
    with open("data/Box Office Manual (2025) - working.txt", "r",encoding="utf-8-sig", errors="ignore") as f:
        return f.read()

manual_context = load_full_manual()

# 3. Simple Reply Function (No Chunks!)
def reply(query: str):
    # We put the manual directly into the system instructions
    sys_instruct = ("You are an expert Symphony Orchestra Box Office Assistant.Below is the complete operations manual. "
                "Use it to provide detailed,step-by-step instructions. If the manual doesn't mention something, "
        "politely say you don't know and suggest asking a Team Leader or supervisor. Do Not use any information listed outside the provided context in your response\n\n"
        f"--- MANUAL START ---\n{manual_context}\n--- MANUAL END ---")
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=query
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