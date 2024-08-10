import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import AgentManager
import logging
from logging_config import setup_logging

db_url = os.getenv("db_url")
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_name = os.getenv("db_name")

# If needed, ensure the logging is set up
setup_logging()

# Create a logger object
logger = logging.getLogger(__name__)


# Initialize the FastAPI application
app = FastAPI(title="Radiologist Agent API-2", version="0.1.0")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConsumerPromptDto(BaseModel):
    """DTO for the end user of the application"""

    message_id: Any
    conversation_id: Any
    consumer_id: Any
    prompt: str


class HumanPromptDto(ConsumerPromptDto):
    """DTO for the human agent"""

    pass  # Inherits all fields from ConsumerPromptDto without additional fields


class ConverseResponseDto(BaseModel):
    """DTO for the response of the conversation endpoint"""

    response: str
    success: bool
    error: str
    type: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]


def extract_tools_name(
    functions: List,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """
    Extracts function names from a list of functions.

    Args:
        functions (List): A list of function objects.

    Returns:
        List[str]: A list of function names.
    """
    tool_name = []
    agent_sources = []
    flag = True
    if functions:
        for function in functions:
            tool_name.append({"action": function.tool_name})
            agent_sources.append(
                {
                    "thought": function.content,
                    "tool_name": function.tool_name,
                    "raw_input": function.raw_input,
                    "raw_output": function.raw_output,
                }
            )
            if function.tool_name != "skip_response_to_the_user":
                flag = False

    return tool_name, agent_sources, flag


def save_history(
    user_message: str, bot_message: str, functions: List[Dict[str, Any]]
) -> None:
    """
    Saves the chat history to a CSV file.

    Args:
        user_message (str): The user's message.
        bot_message (str): The bot's response.
        functions (List): A list of functions used in the chat.
    """
    messages = [user_message]
    types = ["user"]
    tools = [""]

    if functions:
        for function in functions:
            messages.append(function["thought"])
            types.append("function")
            tools.append(function["tool_name"])

    messages.append(bot_message)
    types.append("bot")
    tools.append("")

    # Create a DataFrame from the data
    data = pd.DataFrame({"message": messages, "type": types, "tool": tools})

    # Save the DataFrame to a CSV file
    if os.path.exists("data.csv"):
        data.to_csv("data.csv", mode="a", header=False, index_label="id")
    else:
        data.to_csv("data.csv", mode="w", header=True, index_label="id")

tool_response_dicte = {
    "talk_to_human_agent": "Tell user we are connecting you to a human agent. ",
    "greetings": "Hello, I am Smaro. I can help you with user scheduling, uploading images, and assist with talking to a human agent. "
}

def deal_with_empty(response, tools_names):
    if response == "":
        for tool in tools_names:
            if tool['action'] in ["talk_to_human_agent","greetings"]:
                response = response + tool_response_dicte[tool['action']]
    return response


def handle_message(
    dto: HumanPromptDto,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Handles the message from the user and generates a response using the agent.

    Args:
        dto (HumanPromptDto): The DTO containing the user prompt.

    Returns:
        Tuple[str, List[str]]: The response from the agent and a list of tools used.
    """
    try:
        agent_manager = AgentManager(db_url, db_user, db_password, db_name)
        agent = agent_manager.build_agent(dto.consumer_id, dto.conversation_id)
        logger.info(agent)
        res = agent.chat(dto.prompt)
        tools_names, functions, flag = extract_tools_name(res.sources)
        response = res.response
        #dumb = tools_names
        #if len(dumb) == 0:
        #    dumb = [{"action": "nothing"}]
        response = deal_with_empty(response, tools_names)
        if response == "talk_to_human_agent":
            response = "Tell user we are connecting you to a human agent."

        logger.info(f"\n\n response pre is {response}\n\n")

        if response == "Welcome" or response == "Hello I am Smaro. I can help you with answering frequently asked question, and assist with talking to human agent. ":
            return response, tools_names, functions

        if (
            not response
            or response == "skip_response_to_the_user"
            or response == "None"
            # or flag
            #or dumb[-1]["action"] == "skip_response_to_the_user"
        ):
            response = "Hello. I am Smaro. I can help you with answering frequently asked question, and assist with talking to human agent"
        logger.info(f"response is {response}")
        logger.info(f"tools name are {tools_names}")
        # print(2/"e")
        return response, tools_names, functions
    except Exception as e:
        raise ValueError(f"Error in handle_message function is {e}")


router = APIRouter(
    prefix="/taskproof",
    tags=["llm"],
    responses={404: {"description": "Not found"}},
)


def log_elapsed_time(name: str, st: float, et: float) -> None:
    """
    Logs the elapsed time of a function execution.

    Args:
        name (str): The name of the function.
        st (float): The start time.
        et (float): The end time.
    """
    elapsed_time = et - st
    output = f"{name} Execution time: {'{:.2f}'.format(elapsed_time)} seconds"
    logger.info(output)


@router.post("/converse_faq")
def converse(dto: HumanPromptDto) -> ConverseResponseDto:
    """
    Endpoint to handle conversation requests.

    Args:
        dto (HumanPromptDto): The DTO containing the user prompt.

    Returns:
        ConverseResponseDto: The response from the agent.
    """
    try:
        st = time.time()
        response, tools_names, functions = handle_message(dto)
        et = time.time()
        log_elapsed_time("converse", st, et)
        # save_history(dto.prompt, response, functions)
        return ConverseResponseDto(
            response=response, error="", success=True, type=tools_names, functions=functions
        )
    except Exception as e:
        return ConverseResponseDto(
            response="Sorry, I'm having technical issues. I can help you by answering frequently asked question, and assist with talking to human agent",
            error=str(e),
            success=False,
            type=[],
            functions=[],
        )


# Include the router in the FastAPI application
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 8009)

#python3 main.py
#python main.py

#gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
# uvicorn main:app --host 0.0.0.0 --port 8010  this wins and that looses

# if you are running