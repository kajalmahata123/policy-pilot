from typing import List, BinaryIO
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
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
        self.persist_directory = "chroma_db"

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

    def update_vector_store(self, new_documents: List) -> Chroma:
        """Update or create vector store with new documents"""
        if st.session_state.vector_store is None:
            vector_store = Chroma.from_documents(
                documents=new_documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            vector_store = st.session_state.vector_store
            vector_store.add_documents(new_documents)

        # Persist the changes
        vector_store.persist()
        return vector_store

    @staticmethod
    def get_document_stats(vector_store: Chroma) -> dict:
        """Get statistics about the processed documents"""
        if vector_store is None:
            return {"total_chunks": 0}

        # Get collection stats from Chroma
        total_chunks = vector_store._collection.count()

        return {
            "total_chunks": total_chunks
        }