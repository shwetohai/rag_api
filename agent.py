import ast
import copy
import os
from typing import Any, List, Optional, Type

import mysql.connector
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from llama_index.agent.openai.base import OpenAIAgent
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.llms.openai import OpenAI

from prompts import SYSTEM_PROMPT
from tool import DoctorTool

# Load environment variables
load_dotenv(find_dotenv())


class ToolSpec(BaseModel):
    # this is the type of tool that will be instantiated, it is NOT an instance of the tool
    tool_type: Any


class IntentContext(BaseModel):
    """Context for defining intent and associated tools."""
    tool_specs: Optional[List[ToolSpec]] = []


class AgentManager:
    """Manages chat history retrieval and agent building."""

    def __init__(self, db_url: str, db_user: str, db_password: str, db_name: str, file_path: str = "data.csv"):
        self.db_url = db_url
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.file_path = file_path

    def get_history_from_csv(self) -> List[ChatMessage]:
        """Retrieve chat history from a CSV file."""
        chat_messages: List[ChatMessage] = []
        if os.path.exists(self.file_path):
            df = pd.read_csv(self.file_path)
            types = df["type"].tolist()
            messages = df["message"].tolist()
            tools = df["tool"].tolist()

            for i in range(len(messages)):
                if types[i] == "user":
                    chat_messages.append(ChatMessage(role=MessageRole.USER, content=messages[i]))
                elif types[i] == "function":
                    chat_messages.append(ChatMessage(role=MessageRole.FUNCTION, content=messages[i], additional_kwargs={"name": tools[i]}))
                else:
                    chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=messages[i]))

        return chat_messages[-20:]  # Return the last 20 messages

    def get_history_from_sql(self, consumer_id, chat_conversation_id) -> pd.DataFrame:
        """Retrieve chat history from a SQL database."""
        try:
            connection = mysql.connector.connect(
                host=self.db_url,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name
            )
            print(f"Successfully connected to the database '{self.db_name}'")

            cursor = connection.cursor()
            cursor.execute(f"SELECT chat_conversation_id, user_id, message, type, meta_data, inserted_time FROM chat WHERE chat_conversation_id = {chat_conversation_id} ORDER BY inserted_time")
            records = cursor.fetchall()
            column_names = [i[0] for i in cursor.description]
            df = pd.DataFrame(records, columns=column_names)
        except mysql.connector.Error as e:
            print(f"Error: {e}")
            df = pd.DataFrame()
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                print("Database connection closed")

        return df

    def get_chat_history_from_db(self, consumer_id, chat_conversation_id) -> List[ChatMessage]:
        """Convert SQL database history to list of ChatMessages."""
        try:
            df = self.get_history_from_sql(consumer_id, chat_conversation_id)
            chat_messages: List[ChatMessage] = []

            if not df.empty:
                df["meta_data"] = df["meta_data"].apply(self.safe_literal_eval)
                types = df["type"].tolist()
                messages = df["message"].tolist()
                metadata = df["meta_data"].tolist()

                for i in range(len(messages)):
                    if types[i] == "user":
                        chat_messages.append(ChatMessage(role=MessageRole.USER, content=messages[i].strip('"').strip("'")))
                    elif types[i] == "bot":
                        if metadata[i] and "functions" in metadata[i]:
                            functions = metadata[i]["functions"]
                            for function in functions:
                                chat_messages.append(ChatMessage(role=MessageRole.FUNCTION, content=function["thought"], additional_kwargs={"name": function["tool_name"]}))
                        chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=messages[i].strip('"').strip("'")))

            return chat_messages[-10:-1]
        except Exception as e:
            print(e)
            return []

    def build_agent(self, consumer_id, chat_conversation_id) -> OpenAIAgent:
        """Build and return an OpenAI agent with necessary tools."""
        doctor_tool = ToolSpec(tool_type=DoctorTool)
        all_tools: List[ToolSpec] = [doctor_tool]

        TOOL_TABLE = {
            "unclassified": IntentContext(tool_specs=all_tools),
        }

        available_tools = copy.deepcopy(TOOL_TABLE["unclassified"])

        tools: List[FunctionTool] = []
        for tool_spec in available_tools.tool_specs:
            tool = tool_spec.tool_type()
            tools.extend(tool.to_tool_list())

        llm = OpenAI(model="gpt-4o", temperature=0)
        #chat_history_from_db = self.get_chat_history_from_db(consumer_id, chat_conversation_id)
        #print(f"\n\nchat history is\n {chat_history_from_db}\n\n")
        agent = OpenAIAgent.from_tools(
            tools,
            llm=llm,
            verbose=True,
            system_prompt=SYSTEM_PROMPT,
            #chat_history=chat_history_from_db,
            chat_history=[]
        )
        return agent

    @staticmethod
    def safe_literal_eval(val: str) -> Any:
        """Safely evaluate a string literal."""
        try:
            if isinstance(val, str):
                return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        return None
