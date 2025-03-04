from typing import Dict
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

class LLMManager:
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )

    def create_conversation_memory(self) -> ConversationBufferWindowMemory:
        """Create conversation memory with window buffer"""
        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,  # Remember last 5 interactions
            return_messages=True,
            output_key="answer"
        )

    def create_qa_chain(self, retriever, memory) -> ConversationalRetrievalChain:
        """Create the QA chain with custom prompts"""
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=self.get_condense_prompt(),
            combine_docs_chain_kwargs={"prompt": self.get_qa_prompt()},
            return_source_documents=True,
            verbose=True
        )

    @staticmethod
    def get_condense_prompt() -> PromptTemplate:
        """Get the prompt template for condensing follow-up questions"""
        return PromptTemplate.from_template("""
            Given the following conversation and a follow up question, rephrase the follow up question
            to be a standalone question that captures all relevant context from the chat history.
            If the follow up question is asking for clarification about a previous response,
            make sure to include relevant details from the previous interaction.

            Chat History:
            {chat_history}

            Follow Up Input: {question}
            Standalone Question:
        """)

    @staticmethod
    def get_qa_prompt() -> PromptTemplate:
        """Get the prompt template for question answering"""
        return PromptTemplate.from_template("""
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
