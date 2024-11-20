# from langchain.schema import BaseOutputParser, AgentAction, AgentFinish
from tutor_helper.schema.agent import AgentAction, AgentFinish
from typing import Any, List, Dict, Union
import re
import json
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.agents.agent import AgentOutputParser
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS
from tutor_helper.output_parsers.json import parse_and_check_json_markdown
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.output_parsers import OutputFixingParser
from tutor_helper.common.llms import LlmLoader
from tutor_helper.callbacks.stdout_all import StdOutAllCallbackHandler

# Initialize the logger module
import logging
logger = logging.getLogger(__name__)


class NewAgentOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        logger.debug("text: %s", text)

        try:
            action_match = re.search(
                r"(?:.*)```(?:json|JSON)?\s*(.*?)```", text, re.DOTALL | re.MULTILINE
            )
            if action_match is not None:
                # replace boolean in json
                text2parse = action_match.group(1).strip()
                text2parse = text2parse.replace("True", "true")
                text2parse = text2parse.replace("False", "false")
                # logger.info(text2parse)
                response = json.loads(text2parse, strict=False)
                if isinstance(response, list):
                    # gpt turbo frequently ignores the directive to emit a single action
                    logger.warning("Got multiple action responses: %s", response)
                    response = response[0]
                if response["action"] == "Final Answer":
                    return AgentFinish({"output": response["action_input"]}, text)
                else:
                    return AgentAction(
                        response["action"], response.get("action_input", {}), text
                    )
            else:
                return AgentFinish({"output": text}, text)
        except Exception as e:
            raise OutputParserException(
                f"Could not parse LLM output: {e} {text}"
            ) from e


class NewAgentOutputFixingParser(NewAgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:

        old_parser = NewAgentOutputParser()

        llm_kwargs = dict()
        llm = LlmLoader.create_chat_llm(
            callbacks=[StdOutAllCallbackHandler()],
            **(llm_kwargs or {}),
        )
        parser_ = OutputFixingParser.from_llm(parser=old_parser, llm=llm)
        return parser_.parse(text)


