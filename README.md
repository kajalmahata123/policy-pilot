# Insurance Policy Assistant

A RAG-based chatbot for understanding and querying insurance policy documents using LangChain and GPT-4o.



## Features

- Upload and process PDF and text insurance policy documents
- Natural language querying of policy information
- Contextual conversation with memory
- Source attribution to specific policy documents
- Dynamic retrieval optimization based on query complexity



## Prerequisites

- Python 3.11 or higher
- OpenAI API key

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2Set the OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

To specify a different port:
```bash
streamlit run app.py --server.port=8502
```

## Usage

1. **Upload Documents**:
   - Use the sidebar to upload insurance policy PDF or text files
   - The system will process and index these documents

2. **Ask Questions**:
   - Type your questions about the insurance policies in the chat input
   - The system will retrieve relevant information and provide answers with source citations

3. **View Document Statistics**:
   - Document statistics are displayed in the sidebar
   - You can clear all documents if needed

## Project Structure

- `app.py`: Main Streamlit application
- `document_processor.py`: Handles document loading, chunking, and vectorization
- `rag_engine.py`: Manages the retrieval-augmented generation process
- `llm.py`: Configures the LLM and specialized prompts
- `utils.py`: Helper functions for the UI and data processing
- `styles.css`: Custom CSS styling

## How It Works

1. Documents are processed with LangChain's document loaders and text splitters
2. Text chunks are embedded using OpenAI embeddings and stored in a Chroma vector database
3. User queries are processed through a ConversationalRetrievalChain
4. Relevant document chunks are retrieved using Maximum Marginal Relevance (MMR)
5. GPT-4o generates responses based on the retrieved context and conversation history

## Customization

- Adjust chunk size and overlap in `document_processor.py`
- Modify retrieval parameters in `rag_engine.py`
- Customize prompts in `llm.py`
- Change UI elements in `app.py` and styling in `styles.css`

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Vector storage by [Chroma](https://www.trychroma.com/)
- LLM provided by [OpenAI](https://openai.com/)
- UI powered by [Streamlit](https://streamlit.io/)
