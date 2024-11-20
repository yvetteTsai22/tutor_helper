import concurrent.futures
from typing import Any, Dict, List
from tutor_helper.schema.payload import SearchPayload
from tutor_helper.tools.utilities.llm_utilities import LlmUtilities
from tutor_helper.tools import get_tool_instances_by_config, DEFAULT_TOOLKITS

# Initialize the logger module
import logging
logger = logging.getLogger(__name__)

TOOLS_MAP = DEFAULT_TOOLKITS


class ParallelSearch(object):
    def __init__(self, language="english"):
        self.language = language
        self.max_description_token_count = 250
        self.trim_iter_size = 250

    def _assure_max_doc_description_length(self, docs: List) -> List:
        for doc in docs:
            token_count = LlmUtilities.count_tokens(str(doc["description"]))
            if token_count > self.max_description_token_count:
                # Shorten the description until the max token size it matched
                logger.warning(
                    f"[ParallelSearch]: Description exceeds max token count: {token_count} ({doc['title']})"
                )
                doc["description"] = LlmUtilities.trim_string_to_token_count(
                    str(doc["description"]),
                    self.max_description_token_count,
                    self.trim_iter_size,
                )

        return docs

    def search(self, search_term, tools=None) -> List:
        logger.info(f"Tools: {tools}")
        if tools:
            tools = get_tool_instances_by_config(tools)
        else:
            logger.info(f"Using default tools for {self.language}")
            logger.info(f"Tools: {TOOLS_MAP.get(self.language, [])}")
            tools = [tool() for tool in TOOLS_MAP.get(self.language, [])]

        results = []
        docs = []

        # Create tasks for each tool and gather the results
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_results = [
                executor.submit(self.executor_wrapper, tool, search_term)
                for tool in tools
            ]

            # Collect the results as they complete
            results = [
                future.result()
                for future in concurrent.futures.as_completed(future_results)
            ]

        # Merge results from all tools
        for partial_docs in results:
            docs.extend(partial_docs)

        # Make sure that the document description have a certain MAX length!
        self._assure_max_doc_description_length(docs)


        return docs

    def __call__(self, payload: SearchPayload) -> Any:
        return self.search(payload.query_search, payload.tools)

    def executor_wrapper(self, tool, search_term):

        docs = tool._get_matching_docs(
            search_term,
            tool.num_results
        )
        transformed_docs = tool._transform_docs(docs)

        return transformed_docs
