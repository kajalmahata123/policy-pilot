from typing import Dict, List, Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

class RAGEngine:
    def __init__(self, vector_store):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
        self.vector_store = vector_store
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """Create the QA chain with custom prompts"""

        condense_prompt = PromptTemplate.from_template("""
            Given the following conversation and a follow up question, rephrase the follow up question
            to be a standalone question that captures all relevant context.

            Chat History:
            {chat_history}

            Follow Up Input: {question}
            Standalone Question:
        """)

        qa_prompt = PromptTemplate.from_template("""
            You are an AI assistant specializing in insurance policies. Use the following pieces of 
            context to answer the question at the end. If you don't know the answer, just say that 
            you don't know, don't try to make up an answer. Use the context provided to give detailed 
            and accurate responses.

            Context: {context}

            Question: {question}

            Answer the question in a clear and helpful manner. If you're referencing specific policy 
            details, indicate where this information comes from.
        """)

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 5}
            ),
            memory=self.memory,
            condense_question_prompt=condense_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )

    def process_query(self, query: str) -> Tuple[str, List[Dict]]:
        """Process a query and return the response with sources"""
        if not self.vector_store:
            return "Please upload some documents first.", []

        result = self.qa_chain({"question": query})

        return result["answer"], result["source_documents"]