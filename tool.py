"""Base tool spec class."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from dotenv import find_dotenv, load_dotenv
from llama_index.core.bridge.pydantic import BaseModel, FieldInfo, create_model
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools.tool_spec.base import \
    BaseToolSpec as LlamaBaseToolSpec
from llama_index.core.tools.types import ToolMetadata
from memory import RetrievalChatBot

load_dotenv(find_dotenv())
SPEC_FUNCTION_TYPE = Union[str, Tuple[str, str]]


class BaseToolSpec(LlamaBaseToolSpec):
    """Base tool spec class."""

    FIELD_DESCRIPTIONS = {}

    def __init__(self):
        """Initialize with parameters."""
        self.chutiya = "chutiya"

    def get_fn_schema_from_fn_name(
        self, fn_name: str, spec_functions: Optional[List[SPEC_FUNCTION_TYPE]] = None
    ) -> Optional[Type[BaseModel]]:
        response = super().get_fn_schema_from_fn_name(fn_name, spec_functions)

        # print(f"getting schema for fn_name: {fn_name}")
        # print(f"response schema: {response.schema_json()}")

        if fn_name in self.FIELD_DESCRIPTIONS:
            required_fields = [
                name for name, field in response.__fields__.items() if field.required
            ]
            fields = {
                name: (field.outer_type_, field.default)
                for name, field in response.__fields__.items()
            }

            for field_name, description in self.FIELD_DESCRIPTIONS[fn_name].items():
                if field_name in fields:
                    current_type, current_default = fields[field_name]
                    fields[field_name] = self.field_with_description(
                        current_type, current_default, description
                    )

            response = create_model(fn_name, **fields)

            # Manually set the required fields in the schema after model creation
            schema = response.schema()
            schema["required"] = required_fields

            # If you need the schema to reflect these changes in all subsequent calls:
            response.schema = lambda *args, **kwargs: schema

        # print(f"response schema after: {response.schema_json()}")

        return response

    def field_with_description(self, type_, default_value, description):
        if default_value is ...:
            return (type_, FieldInfo(..., description=description))
        else:
            return (type_, FieldInfo(default_value, description=description))

    def get_metadata_from_fn_name(
        self, fn_name: str, spec_functions: Optional[List[SPEC_FUNCTION_TYPE]] = None
    ) -> Optional[ToolMetadata]:
        """Return map from function name.

        Return type is Optional, meaning that the schema can be None.
        In this case, it's up to the downstream tool implementation to infer the schema.

        """
        try:
            func = getattr(self, fn_name)
        except AttributeError:
            return None
        name = fn_name
        docstring = func.__doc__ or ""
        description = f"{docstring}"
        fn_schema = self.get_fn_schema_from_fn_name(
            fn_name, spec_functions=spec_functions
        )
        return ToolMetadata(name=name, description=description, fn_schema=fn_schema)


class DoctorTool(BaseToolSpec):

    FIELD_DESCRIPTIONS = {
        "answer_frequently_asked_question": {"question": "The question that the user asked"},
        "talk_to_human_agent": {},
        "skip_response_to_the_user": {},
        "greetings": {}
        # "talk_to_human_agent": {"sql_query": "Sql query"}
    }

    spec_functions = [
        "answer_frequently_asked_question",
        "talk_to_human_agent",
        "skip_response_to_the_user",
        "greetings",
    ]

    def answer_frequently_asked_question(self, question: str):
        """
Answers a frequently asked question from the user.
Remember you should not use Markdown formatting in your response to the user, just plain text.
"""
        rag_chatbot = RetrievalChatBot()
        answer = rag_chatbot.run_rag(question)
        return answer
        # return "Here is the image upload popup, please upload the image"


    def talk_to_human_agent(self):
        """
Use this when user wants to talk to a human agent

User message examples:
- I can't log into Smaro app
- I can't create a new template in Smaro
- Unable to raise tickets in Smaro app
- Profile updates not reflecting in Smaro. How to fix?
- Patient reports not showing in panel after adding
- Why can't I raise tickets in Smaro?
- Profile changes not updating in Smaro. Can you help?
- How to ensure patient reports appear in panel?
- Ticket-raising feature not working in Smaro. Can you assist?
- Changes to my profile not reflecting in Smaro
- Reports added not showing in diagnostic panel
- Branches facing login issues with Smaro app
- Trouble creating new templates in Smaro. Can you help?
- Ticket-raising in Smaro app not functioning. Need help.
- Profile changes not showing in Smaro app
- Patient reports not appearing in panel after adding
- The DICOM viewer is showing errors. Troubleshoot?
- The clinical data might be incomplete
- This case is urgent. Provide feedback within the next hour?
"""
        return "Tell user we are connecting you to a human agent. "

    def skip_response_to_the_user(self):
        """
Skip the response to the user

Use this function when you want to skip responding to the user.
This is useful when you don't want to sound repetetive or annoying.
This is also useful when the user sends a message that is not relevant to the conversation or when the user sends a message that is not actionable.
If you exectue this function no message will be sent to the user in response.
"""
        return "SUCCESS. No response will be sent to the user."
    
    def greetings(self):
        """
Use this function when the user greets by saying hello, hi, hey, howdy etc. Never call this function if user is not greeting by saying hello, hi, hello smaro etc 
"""
        return "Hello I am Smaro. I can help you with answering frequently asked question, and assist with talking to human agent."
