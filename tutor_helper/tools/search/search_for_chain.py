from langchain.tools import StructuredTool
from tutor_helper.chains.knowledge_research import (
    KnowledgeResearch,
)
from pydantic import BaseModel


def knowledge_research(
    similarity_search_term: str,
    request_raw_question_input: str,
    notes: str = "",
):
    """Researches and returns content from Trend Micro knowledge base articles and technical product documentation."""

    internal_knowledge_research_chain = KnowledgeResearch()
    answer = internal_knowledge_research_chain(
        {
            "description": similarity_search_term,
            "revised_question": request_raw_question_input,
            "notes": notes,
        }
    )

    return answer


def knowledge_research_tool():
    return StructuredTool.from_function(knowledge_research)
