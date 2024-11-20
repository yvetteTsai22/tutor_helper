"""Toolkit for drafting KB articles."""
from __future__ import annotations
from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool

from tutor_helper.tools.search.search_for_chain import (
    knowledge_research_tool
)


class SimplifiedToolkit(BaseToolkit):
    """Tools for looking up acronyms, debugging and troubleshooting procedures."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            # Search tools
            knowledge_research_tool(),
        ]
