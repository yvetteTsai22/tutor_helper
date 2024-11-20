from langchain.output_parsers import StructuredOutputParser as StructuredOutputParser_
from tutor_helper.output_parsers.json import parse_and_check_json_markdown
from typing import Any


class StructuredOutputParser(StructuredOutputParser_):
    def parse(self, text: str) -> Any:
        expected_keys = [rs.name for rs in self.response_schemas]
        return parse_and_check_json_markdown(text, expected_keys)

    def get_default_response(self):
        # to get the default response based on the response schemas
        return {rs.name: None for rs in self.response_schemas}
