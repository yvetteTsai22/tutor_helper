from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from tutor_helper.chains.extract_and_combine import ExtractAndCombine
from tutor_helper.chains.search_tools_parallel import SearchToolsParallel

from tutor_helper.common.llms import LlmLoader


from tutor_helper.tools.search.search import (
    DuckDuckGoSearch
)

import re
from langchain.tools import StructuredTool

import logging 
logger = logging.getLogger(__name__)


class KnowledgeResearch(Chain):
    """TS Initial Response"""

    tools:List = [DuckDuckGoSearch()]
    name:str = "knowledge_research"

    input_variables: List[str] = [
        "description",
        "notes",
    ]
    output_variables: List[str] = ["content", "references"]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return self.output_variables

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        # Engineer notes
        if inputs["notes"] is not None:
            # Extracting ids from the notes [hash format or KB article id]
            notes = re.findall(
                r"([a-fA-F\d]{32})|(\d{9})", inputs["notes"]
            )
            notes = [
                item[0] or item[1] for item in notes if item[0] or item[1]
            ]
        else:
            notes = []

        # Searching for docs
        # LLM | Getting the search term
        # search_term_tool = SearchTerm()
        # search_term = search_term_tool.from_description(inputs["description"])
        search_term = inputs["description"]
        logger.info(f"[chains.ts.int_research._call] - Search term: {search_term}")

        # LLM | Getting search results from tools (LLM used behind to choose & select)
        search_tools_parallel_chain = SearchToolsParallel(self.tools)
        raw_docs = search_tools_parallel_chain.run(
            search_term, [], notes
        )
        print(raw_docs)
        # Splitting documents which content is > than XXXX tokens
        # Create multiple documents with the same metadata and split content
        # This is to avoid the 2048 tokens limit of the LLM
        raw_docs = search_tools_parallel_chain.chunk(raw_docs)

        # Transform docs to the format expected by the chain
        docs = search_tools_parallel_chain.transform(raw_docs)

        print("Num docs to summarize", len(docs))

        logger.info(
            f"[chains.ts.int_research._call] - Found raw_docs: {len(raw_docs)}"
        )
        logger.info(f"[chains.ts.int_research._call] - Found docs: {len(docs)}")

        ## Creating Chat version of the 3.5
        llm = LlmLoader.create_chat_llm(
            model=LlmLoader.DEPLOYMENT_35_TURBO,
            verbose=True,
            request_timeout_seconds=60
        )

        # Extract and summarize the content
        extractAndCombine = ExtractAndCombine(llm)
        # Default response here to avoid error
        response_json = extractAndCombine.output_parser.get_default_response()
        try:
            response_json = extractAndCombine.run(
                docs=docs,
                search_term=search_term,
                description=inputs["revised_question"],
                tools=self.tools,
                docs_by_id=notes,
            )
        except Exception as e:
            logger.error(f"Error running ExtractAndCombine: {e}")

        logger.info(response_json)

        return response_json

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "ts_initial_response"
