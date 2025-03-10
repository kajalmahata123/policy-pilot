
"""
A class for processing and managing document uploads, vectorization, and storage.
This class handles the processing of PDF and text documents, splitting them into manageable chunks,
creating embeddings using OpenAI, and storing them in a Chroma vector database. It also manages
the persistence of the vector store and provides document statistics.
Attributes:
    text_splitter (RecursiveCharacterTextSplitter): Splits text into chunks with specified parameters.
    embeddings (OpenAIEmbeddings): OpenAI embeddings model for vectorizing text.
    persist_directory (str): Directory path where the Chroma database is stored.
Methods:
    process_file(uploaded_file: BinaryIO) -> List:
        Processes a single uploaded file (PDF or text) and returns split documents.
    update_vector_store(new_documents: List) -> Chroma:
        Updates the vector store with new documents or creates a new one.
    get_document_stats(vector_store: Chroma) -> dict:
        Returns statistics about the processed documents in the vector store.
"""
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

        # Try to load existing vector store
        if os.path.exists(self.persist_directory) and st.session_state.vector_store is None:
            try:
                st.session_state.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"Error loading existing vector store: {e}")
                st.session_state.vector_store = None

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