from typing import Dict, List, Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

class RAGEngine:
    def __init__(self, vector_store):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
        self.vector_store = vector_store
        # Enhanced memory with window buffer and summary
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,  # Remember last 5 interactions
            return_messages=True,
            output_key="answer"
        )

        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """Create the QA chain with custom prompts"""

        condense_prompt = PromptTemplate.from_template("""
            Given the following conversation and a follow up question, rephrase the follow up question
            to be a standalone question that captures all relevant context from the chat history.
            If the follow up question is asking for clarification about a previous response,
            make sure to include relevant details from the previous interaction.

            Chat History:
            {chat_history}

            Follow Up Input: {question}
            Standalone Question:
        """)

        qa_prompt = PromptTemplate.from_template("""
            You are an AI assistant specializing in insurance policies. Use the following pieces of 
            context to answer the question at the end. If you don't know the answer, just say that 
            you don't know, don't try to make up an answer.

            When answering:
            1. If this is a follow-up question, reference relevant information from previous responses
            2. Be specific about which parts of the policy you're referencing
            3. If there are related topics that might be helpful, mention them briefly
            4. If you need clarification, ask specific follow-up questions

            Context: {context}

            Current Question: {question}

            Chat History:
            {chat_history}

            Answer the question in a clear and helpful manner. If you're referencing specific policy 
            details, indicate where this information comes from.
        """)

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs=self._get_mmr_search_params("placeholder") #initial call to get default params
            ),
            memory=self.memory,
            condense_question_prompt=condense_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=True
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