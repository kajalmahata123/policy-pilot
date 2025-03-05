
"""
A Streamlit-based insurance policy assistant application that processes and analyzes insurance documents.

This application provides a user interface for:
- Uploading insurance policy documents (PDF or TXT)
- Processing and vectorizing document content
- Interactive chat interface to query insurance policy information
- Display of document statistics and chat history

The app uses RAG (Retrieval-Augmented Generation) to provide accurate responses based on the uploaded documents.

Dependencies:
    - streamlit
    - document_processor module
    - rag_engine module
    - utils module (init_session_state, load_css, display_chat_history, validate_api_key, format_sources)

Session State Variables:
    - vector_store: Stores the vectorized document content
    - uploaded_files: List of uploaded document files
    - chat_history: List of user-assistant interactions

Returns:
    None

Note:
    Requires valid API key configuration to function properly.
"""
import streamlit as st
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from utils import (
    init_session_state,
    load_css,
    display_chat_history,
    validate_api_key,
    format_sources
)

def main():
    st.set_page_config(
        page_title="Insurance Policy Assistant",
        page_icon="📋",
        layout="wide"
    )

    load_css()
    init_session_state()

    if not validate_api_key():
        return

    st.title("📋 Insurance Policy Assistant")

    # Initialize processors
    doc_processor = DocumentProcessor()

    # Sidebar for file upload and document status
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload insurance policy documents",
            accept_multiple_files=True,
            type=["pdf", "txt"],
            help="Upload PDF or text files containing insurance policy documents"
        )

        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_splits = []
                for file in uploaded_files:
                    if file.name not in [f.name for f in st.session_state.uploaded_files]:
                        splits = doc_processor.process_file(file)
                        all_splits.extend(splits)

                if all_splits:
                    st.session_state.vector_store = doc_processor.update_vector_store(all_splits)
                    st.session_state.uploaded_files = uploaded_files

        # Display document statistics
        if st.session_state.vector_store:
            stats = doc_processor.get_document_stats(st.session_state.vector_store)
            st.markdown("### Document Statistics")
            st.markdown(f"Total chunks: {stats['total_chunks']}")

            if st.button("Clear All Documents"):
                if st.session_state.vector_store:
                    # Properly close and delete Chroma collection
                    st.session_state.vector_store._client.reset()
                st.session_state.vector_store = None
                st.session_state.uploaded_files = []
                st.session_state.chat_history = []
                st.rerun()

    # Main chat interface
    if st.session_state.vector_store:
        rag_engine = RAGEngine(st.session_state.vector_store)

        # Display chat history
        display_chat_history()

        # Query input
        query = st.chat_input("Ask a question about your insurance policies")

        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})

            with st.spinner("Thinking..."):
                response, sources = rag_engine.process_query(query)

                message = {
                    "role": "assistant",
                    "content": response,
                    "sources": format_sources(sources)
                }
                st.session_state.chat_history.append(message)

                # Force a rerun to update the chat display
                st.rerun()
    else:
        st.info("👈 Please upload some insurance policy documents to get started!")

if __name__ == "__main__":
    main()