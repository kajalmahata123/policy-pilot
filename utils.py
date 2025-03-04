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
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = {
            'current_topic': None,
            'last_reference': None,
            'follow_up_suggestions': []
        }

def load_css():
    """Load custom CSS"""
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def display_chat_history():
    """Display chat history with proper formatting"""
    for i, message in enumerate(st.session_state.chat_history):
        role = message["role"]
        content = message["content"]

        # Add visual grouping for related messages
        if i > 0 and role == st.session_state.chat_history[i-1]["role"]:
            continuation_class = "message-continuation"
        else:
            continuation_class = ""

        if role == "user":
            st.markdown(f"""
                <div class="chat-message user-message {continuation_class}">
                    <b>You:</b> {content}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message assistant-message {continuation_class}">
                    <b>Assistant:</b> {content}
                </div>
                """, unsafe_allow_html=True)

            if "sources" in message:
                st.markdown(f"""
                    <div class="source-reference">
                        Sources: {message['sources']}
                    </div>
                    """, unsafe_allow_html=True)

            if "suggestions" in message:
                st.markdown("""
                    <div class="follow-up-suggestions">
                        <b>Related questions you might want to ask:</b>
                    </div>
                    """, unsafe_allow_html=True)
                for suggestion in message["suggestions"]:
                    st.button(
                        suggestion,
                        key=f"suggestion_{i}_{suggestion}",
                        help="Click to ask this follow-up question"
                    )

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

def update_conversation_context(query: str, response: str, sources: List[Dict]):
    """Update the conversation context based on the latest interaction"""
    if 'conversation_context' not in st.session_state:
        init_session_state()

    # Extract potential topics from the response
    topics = [doc.metadata.get('source', '').split('.')[0] for doc in sources if hasattr(doc, 'metadata')]
    if topics:
        st.session_state.conversation_context['current_topic'] = topics[0]

    # Store the last reference for potential follow-ups
    if sources:
        st.session_state.conversation_context['last_reference'] = sources[0]