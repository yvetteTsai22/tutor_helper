from langchain.docstore.document import Document
from langchain.output_parsers import StructuredOutputParser as StructuredOutputParser_
from langchain.output_parsers import ResponseSchema
from langchain.tools import BaseTool


from tutor_helper.common.llms import LlmLoader

from tutor_helper.output_parsers.json import parse_and_check_json_markdown
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import os
from langchain.tools.base import create_schema_from_function
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging
logger = logging.getLogger(__name__)

DEFALT_LANGUAGE = os.getenv("DEFALT_LANGUAGE", "English")


class StructuredOutputParser(StructuredOutputParser_):
    def parse(self, text: str) -> Any:
        logger.info(text)
        logger.info(type(text))
        expected_keys = [rs.name for rs in self.response_schemas]
        logger.info(self.response_schemas)
        return parse_and_check_json_markdown(text, expected_keys)


class DocumentPickerTool(BaseTool, ABC):
    id_key:str = "url" # unique id of a doc
    display_name:str = "Document"
    # to determine whether to display the result
    is_displayable:bool = True
    index_fields:dict = {
        "title": "title",
        "description":"description"
    }
    
    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"

    @property
    def args(self) -> dict:
        return create_schema_from_function(
            model_name=f"{self.name}Schema", func=self._run_query
        ).model_json_schema()["properties"]
    
    @classmethod
    def initialize_formats(cls):
        cls.input_schemas = [
            ResponseSchema(
                name="description",
                description="description content to be used for the search",
            )
        ]
        cls.input_parser = StructuredOutputParser.from_response_schemas(
            cls.input_schemas
        )
        cls.input_format = cls.input_parser.get_format_instructions()

        cls.output_schemas = [
            ResponseSchema(
                name="docs",
                description="list of matching docs",
            )
        ]
        cls.output_parser = StructuredOutputParser.from_response_schemas(
            cls.output_schemas
        )
        cls.output_format = cls.output_parser.get_format_instructions()

    @abstractmethod
    def _get_matching_docs(
        self, search_term: str, num_results: int,  **kwargs
    ) -> List[Document]:
        pass
    
    @abstractmethod
    def _transform_docs(self, docs: List) -> List:
        pass
    
    def _get_related_doc_ids(self, description: str, docs: List[Document]) -> List:
        prompt = """
You are a tutor assistant.
Your task is to provide the list of document ID's that might be related to the provided description.
DESCRIPTION: {description}
DOCS
---
{formated_docs}
---
EXPECTED ANSWER FORMAT: list of document ids comma separated (e.g. fd7e1b4893e0453f3412bc45bbf697ed,1ea9b5ad6a32192e80ebe9ce5b7b4b89). Reply empty if no document is related.
        """

        # Remove duplicated based on doc.metadata[self.id_key]
        docs = list({doc.metadata[self.id_key]: doc for doc in docs}.values())

        formated_docs = "\n".join(
            [
                f"{doc.metadata[self.id_key]}: [{doc.metadata[self.index_fields['title']]}] "
                + doc.metadata[self.index_fields['description']].replace("\n", " ")
                for doc in docs
            ]
        )

        logger.info(f"TOOL: {self.display_name}")
        logger.info(f"FORMATED DOCS:\n{formated_docs}")

        llm = LlmLoader.create_chat_llm(model=LlmLoader.DEPLOYMENT_35_TURBO)
        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt),
        )

        selected_ids = chain(
            {"description": description, "formated_docs": formated_docs}
        )["text"]

        return [x.strip() for x in selected_ids.split(",")]
    
    def _run(self, query) -> str:
        # Parsing input query/product json
        logger.info(f"input: {query}")
        request = self.input_parser.parse(query)

        search_term = request["description"]
        language = request.get("language", DEFALT_LANGUAGE)

        # Getting top N matching documents
        matching_docs = self._get_matching_docs(
            search_term, self.num_results
        )

        if len(matching_docs) == 0:
            selected_ids = []
        else:
            logger.info(f"matching_docs: {matching_docs}")
            selected_ids = self._get_related_doc_ids(
                request["description"], matching_docs
            )

        selected_docs = [
            doc
            for doc in matching_docs
            if doc.metadata[self.id_key] in selected_ids
        ]

        # Setting source to the DOC id to be used at the "references" section
        # Adding pulled tool metadata
        transformed_docs = self._transform_docs(selected_docs)

        return '```json{"docs":' + json.dumps(transformed_docs) + "}```"
    
    def from_description(self, description: str) -> List[Dict]:
        docs = self._get_matching_docs(description, self.num_results)
        docs =  self._transform_docs(docs)
        docs = [dict(i,**{"id": i[self.id_key], "content": i["description"]}) for i in docs]
        return docs
