from langchain.tools import BaseTool
from .contracts.document_picker import DocumentPickerTool
from typing import List, Dict, Any, Union
from .search.search import DuckDuckGoSearch
import logging
logger = logging.getLogger(__name__)
TOOL_MAPPING = {
    name: cls
    for name, cls in globals().items()
    if isinstance(cls, type) and issubclass(cls, DocumentPickerTool)
}

DEFAULT_TOOLKITS = {
    "english": [
        DuckDuckGoSearch
    ],
    "japanese": [
    ],
    # Add more languages here as needed
}


def get_tools_by_name(names: List[str]) -> List:

    return [TOOL_MAPPING[name] for name in names if name in TOOL_MAPPING]


def get_tool_instances_by_config(names: List[Union[str, dict]]) -> List[BaseTool]:
    tools = []
    for i in names:
        if isinstance(i, str):
            logger.debug(f"initializing tool: {i}")
            tools.append(TOOL_MAPPING[i]())
        elif isinstance(i, dict):
            tool = TOOL_MAPPING[i["name"]](**i["config"])
            tools.append(tool)
        else:
            logger.error(f"Invalid tool config: {i}")

    return tools
