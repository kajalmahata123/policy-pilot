from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from llm import LLMManager

class RAGEngine:
    """
    A class to handle retrieval-augmented generation (RAG) for question answering.

    Attributes:
        vector_store: The vector store used for document retrieval.
        llm_manager: The manager for the language model.
        memory: The conversation memory for the QA chain.
        qa_chain: The question-answering chain with custom prompts.
    """

    def __init__(self, vector_store):
        """
        Initializes the RAGEngine with a vector store and sets up the QA chain.

        Args:
            vector_store: The vector store used for document retrieval.
        """
        self.vector_store = vector_store
        self.llm_manager = LLMManager()
        self.memory = self.llm_manager.create_conversation_memory()
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        """
        Creates the QA chain with custom prompts.

        Returns:
            The QA chain object.
        """
        return self.llm_manager.create_qa_chain(
            retriever=self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs=self._get_mmr_search_params("placeholder")
            ),
            memory=self.memory
        )

    def process_query(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Processes a query and returns the response with sources.

        Args:
            query: The query string to process.

        Returns:
            A tuple containing the answer string and a list of source documents.
        """
        if not self.vector_store:
            return "Please upload some documents first.", []

        # Update MMR parameters based on the current query
        self.qa_chain.retriever.search_kwargs = self._get_mmr_search_params(query)

        # Get the result from the chain
        result = self.qa_chain({"question": query})

        return result["answer"], result["source_documents"]

    def get_chat_history(self) -> List[Dict]:
        """
        Retrieves the current chat history.

        Returns:
            A list of dictionaries representing the chat history.
        """
        messages = self.memory.chat_memory.messages
        history = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})

        return history

    def _get_mmr_search_params(self, query: str) -> Dict:
        """
        Dynamically adjusts MMR search parameters based on query characteristics.

        Args:
            query: The query string to analyze.

        Returns:
            A dictionary of MMR search parameters.
        """
        # Default parameters
        params = {
            "k": 3,
            "fetch_k": 5,
            "lambda_mult": 0.7,
            "filter_similarity": 0.85
        }

        # Adjust for question complexity
        if len(query.split()) > 15:  # Complex question
            params.update({
                "k": 4,
                "fetch_k": 8,
                "lambda_mult": 0.6  # Increase diversity for complex questions
            })
        elif "compare" in query.lower() or "difference" in query.lower():
            params.update({
                "k": 4,
                "fetch_k": 6,
                "lambda_mult": 0.5  # Maximum diversity for comparison questions
            })

        return params