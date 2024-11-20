# Schema defined for FASTAPI request and common chains, agents and playbooks
from pydantic import BaseModel, Field, computed_field
from typing import Any, Optional, List, Union
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.output_parsers import ResponseSchema
from enum import Enum
import json
from abc import ABC, abstractmethod


class MessageType(Enum):
    PROMPT = "prompt"
    SYSTEM_MESSAGE = "system_message"
    HUMAN_MESSAGE = "human_message"
    AI_MESSAGE = "ai_message"


class Message(BaseModel):
    type: MessageType
    content: str


class Tool(BaseModel):
    name: str
    config: dict


# TODO: should be an attribute of MessageType
message_type_to_template_class = {
    MessageType.PROMPT: HumanMessagePromptTemplate,
    MessageType.SYSTEM_MESSAGE: SystemMessagePromptTemplate,
    MessageType.HUMAN_MESSAGE: HumanMessagePromptTemplate,
    MessageType.AI_MESSAGE: AIMessagePromptTemplate,
    # Add other mappings here if needed
}


class DefaultPayload(BaseModel):
    language: Optional[str] = Field(
        "",
        description="Language given in payload, will overwrite the language detected by the system",
    )
    scope_meta: Optional[str] = Field(
        "",
        description="scope meta, containing the language of the payload or others later..",
    )

    @property
    def text(self):
        """A text property to be used for language detection

        Raises:
            NotImplementedError: ask each payload to decide
        """
        raise NotImplementedError


class SearchTermPayload(BaseModel):
    description: str

    @property
    def text(self):
        return self.description

class TaskPayloadNew(DefaultPayload):
    documents: List
    task_action: Optional[str] = Field(
        None,
        description="Optional. When f.e. JP needs an english prompt for certain tasks",
    )
    messages: List[Message] = Field(
        [], description="List of messages, system messages, human messages"
    )
    response_schemas: List = Field(
        [], description="List of response schemas, to set the format of the response"
    )

    @property
    def text(self):
        # FIXME: which field to detect language?
        return self.scope_meta

    class Config:
        extra = "allow"


class SearchPayload(DefaultPayload):
    query_search: str
    tools: Optional[List[Union[str, dict]]] = Field(
        None, description="Tools to use in search, None for default"
    )

    @property
    def text(self):
        return self.query_search


class SuggestPayload(BaseModel):
    description: str
    documents: List

    @property
    def text(self):
        return self.description


class AgentPayload(BaseModel):
    similarity_search_term: str
    request_raw_question_input: str
    products: Optional[List[str]] = Field(
        [], description="Products related to the question"
    )


class GenerateSearchTermAndSearchPayload(DefaultPayload):

    question: str
    product: Optional[str] = Field(None, description="Products related to the question")
    product_version: Optional[str] = Field(None, description="Product version")
    os: Optional[str] = Field(None, description="OS")
    tools: Optional[List[Union[str, Tool]]] = Field(
        None, description="Tools to use in search, None for default"
    )

    @property
    def text(self):
        return self.subject


def get_response_schema_by_class(data_model: BaseModel) -> list:
    schema = []
    for k, v in data_model.model_json_schema()["properties"].items():
        schema.append(
            {
                "name": k,
                "description": v.get("description"),
                "type": v.get("type") if v.get("type") else "string",
            }
        )
    return schema

