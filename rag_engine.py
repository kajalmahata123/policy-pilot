from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from llm import LLMManager

class RAGEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm_manager = LLMManager()
        self.memory = self.llm_manager.create_conversation_memory()
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        """Create the QA chain with custom prompts"""
        return self.llm_manager.create_qa_chain(
            retriever=self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs=self._get_mmr_search_params("placeholder")
            ),
            memory=self.memory
        )

    def process_query(self, query: str) -> Tuple[str, List[Dict]]:
        """Process a query and return the response with sources"""
        if not self.vector_store:
            return "Please upload some documents first.", []

        # Update MMR parameters based on the current query
        self.qa_chain.retriever.search_kwargs = self._get_mmr_search_params(query)

        # Get the result from the chain
        result = self.qa_chain({"question": query})

        return result["answer"], result["source_documents"]

    def get_chat_history(self) -> List[Dict]:
        """Get the current chat history"""
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
        Dynamically adjust MMR search parameters based on query characteristics
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