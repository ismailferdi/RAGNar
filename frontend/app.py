import os
import streamlit as st
import httpx
from frontend.components import uploader, chat

st.set_page_config(page_title="Ragnar – Document Q&A", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "documents" not in st.session_state:
    st.session_state["documents"] = []

API_BASE_URL = os.getenv("RAGNAR_API_URL", "http://localhost:8000")

try:
    response = httpx.get(f"{API_BASE_URL}/documents/")
    response.raise_for_status()
    st.session_state["documents"] = response.json()
except httpx.RequestError as e:
    st.warning(
        "Failed to fetch documents from the backend. Some features may be limited."
    )
    st.session_state["documents"] = []


col_left, col_right = st.columns([1, 2])

with col_left:
    uploader.render()


with col_right:
    chat.render()
