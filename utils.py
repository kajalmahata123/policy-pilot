import os
from typing import List, Dict, Any
import streamlit as st

def init_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def load_css():
    """Load custom CSS"""
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def display_chat_history():
    """Display chat history with proper formatting"""
    for i, message in enumerate(st.session_state.chat_history):
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <b>You:</b> {content}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <b>Assistant:</b> {content}
                </div>
                """, unsafe_allow_html=True)
            
            if "sources" in message:
                st.markdown(f"""
                    <div class="source-reference">
                        Sources: {message['sources']}
                    </div>
                    """, unsafe_allow_html=True)

def validate_api_key() -> bool:
    """Validate that OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set the OPENAI_API_KEY environment variable.")
        return False
    return True

def format_sources(source_documents: List[Dict[str, Any]]) -> str:
    """Format source documents for display"""
    sources = []
    for doc in source_documents:
        if hasattr(doc, 'metadata'):
            page_info = f"(Page {doc.metadata.get('page', 'N/A')})" if 'page' in doc.metadata else ""
            source = f"{doc.metadata.get('source', 'Unknown')} {page_info}"
            sources.append(source)
    return "; ".join(sources)
