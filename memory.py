import os
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Load environment variables from .env file
load_dotenv()
class RetrievalChatBot:
    def __init__(self, persist_directory: str = "./shwet_rag_db3"):
        self.persist_directory = persist_directory
        self.vector_store = self._load_embeddings_chroma()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        #self.memory = self._get_memory()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def _load_embeddings_chroma(self) -> Chroma:
        """
        Load a Chroma vector store with the specified embedding function.
        Returns:
            Chroma: The loaded Chroma vector store.
        """
        #embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = Chroma(
            collection_name="example_collection", persist_directory=self.persist_directory, embedding_function=embeddings
        )
        return vector_store
    
    def _get_memory(self) -> ConversationBufferMemory:
        """
        Retrieve the conversation memory from a CSV file, if it exists.
        Returns:
            ConversationBufferMemory: The conversation memory.
        """
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        if os.path.exists("memory.csv"):
            df = pd.read_csv("memory.csv")
            messages = df["message"].tolist()
            chat_messages = [{"question": msg["user"], "answer": msg["bot"]} for msg in messages]
            chat_messages = chat_messages[-20:]
            for chat in chat_messages:
                memory.save_context({"input": chat["question"]}, {"output": chat["answer"]})
        return memory
    
    def ask_question(self, question: str, chain: ConversationalRetrievalChain) -> str:
        """
        Ask a question using the provided conversational retrieval chain.
        Args:
            question (str): The question to ask.
            chain (ConversationalRetrievalChain): The conversational retrieval chain.
        Returns:
            str: The answer from the chain.
        """
        result = chain.invoke({"question": question})
        return result["answer"]
    
    def run_rag(self, query: str) -> str:
        """
        Run the retrieval-augmented generation process with the given query.
        Args:
            query (str): The query to run.
        Returns:
            str: The result of the RAG process.
        """
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        system_template = r"""
        Use the following pieces of context to answer the user's question in maximum 40 words.
        If you don't find the answer in the provided context, just respond "I don't know."
        ---------------
        Context: ```{context}```
        """

        user_template = """
        Question: ```{question}```
        """
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template),
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)
        crc = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=True,
        )
        result = self.ask_question(query, crc)
        return result
