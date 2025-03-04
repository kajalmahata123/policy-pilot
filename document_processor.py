from typing import List, BinaryIO
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import tempfile

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )
        self.embeddings = OpenAIEmbeddings()

    def process_file(self, uploaded_file: BinaryIO) -> List:
        """Process a single uploaded file"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            if uploaded_file.name.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = TextLoader(tmp_file_path)

            documents = loader.load()

            # Add metadata
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            # Split documents
            splits = self.text_splitter.split_documents(documents)

            return splits
        finally:
            os.unlink(tmp_file_path)

    def update_vector_store(self, new_documents: List) -> FAISS:
        """Update or create vector store with new documents"""
        if st.session_state.vector_store is None:
            vector_store = FAISS.from_documents(new_documents, self.embeddings)
        else:
            current_store = st.session_state.vector_store
            vector_store = current_store.add_documents(new_documents)

        return vector_store

    @staticmethod
    def get_document_stats(vector_store: FAISS) -> dict:
        """Get statistics about the processed documents"""
        if vector_store is None:
            return {"total_chunks": 0}

        # Get the total number of documents using the index to docstore mapping
        total_chunks = len(vector_store.index_to_docstore_id)

        return {
            "total_chunks": total_chunks
        }