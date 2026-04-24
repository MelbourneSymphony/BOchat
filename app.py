import streamlit as st
from google import genai
import time
from google.api_core import exceptions
import os

# 1. Setup Gemini
gemini_token = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=gemini_token)

# 2. Load the entire manual
@st.cache_data
def load_full_manual():
    with open("data/Box Office Manual (2025) - working.txt", "r",encoding="utf-8-sig", errors="ignore") as f:
        return f.read()

manual_context = load_full_manual()

# 3. Reply Function
def reply(query: str):
    #manual directly into the system instructions
    sys_instruct = ("You are an expert Symphony Orchestra Box Office Assistant.Below is the complete operations manual. "
                "Use it to provide detailed,step-by-step instructions. If the manual doesn't mention something, "
        "politely say you don't know and suggest asking a Team Leader or supervisor. Do Not use any information listed outside the provided context in your response\n\n"
        f"--- MANUAL START ---\n{manual_context}\n--- MANUAL END ---")
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config={'system_instruction': sys_instruct},
                contents=query
            )
            return response.text
        except exceptions.ServiceUnavailable:
            if attempt < 2:
                time.sleep(2)  # Wait 2 seconds before trying again
                continue
            else:
                return "The server is a bit busy right now. Please try your question again in a moment!"
#Streamlit UI settings
st.title("BObot")

# Initialize chat history with a welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I assist you with Box Office operations today?"}
    ]

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
if query := st.chat_input("Ask about ticket exchanges, refunds, or seating..."):
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Checking the manual..."):
            try:
                answer = reply(query)
                # Escape $ to avoid LaTeX issues in Streamlit
                answer = answer.replace("$", "\$")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Server is busy or an error occurred: {e}")