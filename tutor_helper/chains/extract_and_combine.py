from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from typing import Any, Dict, List, Optional
from langchain.output_parsers import ResponseSchema
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import concurrent.futures

from langchain.chains.summarize import load_summarize_chain

from tutor_helper.tools.utilities.utils import num_tokens_from_string, split_by_token
from tutor_helper.output_parsers.structured import StructuredOutputParser

import logging 
logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """You are an experienced tutor.
Return the title and content of the document if it is relevant to the question and remove non relevant content.
=========
```{context}```
=========
QUESTION: ```{question}```
If the relevant text is the entire document, return all of it's content and title.
Extract the relevant content to answer the question, keeping notes, details and howto access/run actions.
Make the content as concise as possible.
If document is not related ... return DOCUMENT NOT RELATED (<reason why is not related or relevant>)."""

COMBINE_PROMPT = """You are an experienced tutor.
Your task is to combine extracted parts of a long document for further processing. All given documents are related to the question.
Include recommendation notes if available, along with the reasons behind them. Keeping details and howto access/run actions.
Remove duplicate content.
Only use content provided by the tools.

QUESTION: ```{question}```
=========
```{summaries}```
=========
Your response should be in markdown format.
At the bottom of the response you should include #References section with a list of documents and their ID's used to compose your answer.
Also indicate in percent how much each document contributed to the answer.
Sample References section:
# References (percent contribution)
[ID1 (10%),ID2 (45%), ID3 (5%), ....]
"""

SYSTEM_PROMPT = """Your task is to return the following knowledge content,
be aware that below knowledge details might contain additional or irrelevant information not related to the question scope/implementation,
these details were collected from internal knowledge articles, that might not be 100% related to the question.
"""

HUMAN_PROMPT = """Return the knowledge content in minimized HTML with inline HTML elements styling.
Replace/Remove competitor or other vendors names/instructions if not part of the question.

Do not include documents in the references that were not used in the answer.

KNOWLEDGE CONTENT:
```{research_summary}```

QUESTION:
```{question}```

FORMAT INSTRUCTIONS:
{format_instructions}

IMPORTANT: If "KNOWLEDGE CONTENT" is empty, just indicate no info found. Don't try to make up an answer.
"""


class ExtractAndCombine:

    response_schemas = [
        ResponseSchema(
            name="content",
            description="Internal knowledge information in HTML format with inline css style",
        ),
        ResponseSchema(
            name="references",
            description="A list of document/reference IDs which were used (ordered by contribution).",
        ),
        ResponseSchema(
            name="response_outcome",
            description="Boolean true/false if the question was answered.",
        ),
        ResponseSchema(
            name="response_rating",
            description="Value from 0 - 10. 0 being unanswered and 10 being fully answered.",
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    def __init__(
        self,
        llm,
        extract_prompt: str = EXTRACT_PROMPT,
        combine_prompt: str = COMBINE_PROMPT,
    ):
        self.llm = llm

        self.extract_prompt = PromptTemplate(
            template=extract_prompt,
            input_variables=["context", "question"],
        )

        self.combine_prompt = PromptTemplate(
            template=combine_prompt,
            input_variables=["summaries", "question"],
            # partial_variables = {"format_instructions": self.format_instructions},
        )

    def run(
        self,
        docs: List,
        search_term: str,
        description: str,
        tools: List,
        docs_by_id: List = [],
    ):
        """Give a list of searched docs, extract the relevant content and combine it.

        Args:
            docs (List): A list of documents to extract from.
            search_term (str): the search term of user's inquiry
            description (str): the full description of user's inquiry
            tools (List): the tool that extracted the Documents, to format and detremine whether to display the reference
            docs_by_id (List, optional): docs that were given by IDs. Defaults to [].

        Returns:
            _type_: _description_
        """
        # Creating extract chain
        extract_chain = LLMChain(
            verbose=True,
            llm=self.llm,
            prompt=self.extract_prompt,
        )
        logger.info(f"tools: {[tool.__str__() for tool in tools]}")

        ## Running summaries parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Define a function to be executed by ThreadPoolExecutor
            def extract_task(doc, search_term):
                
                task_response = doc.metadata["source"], extract_chain.run(
                    context=doc.page_content
                    + "\nVISIBILITY:"
                    + doc.metadata["visibility"],
                    question=search_term,
                )

                return task_response

            # Submit the tasks to the executor
            future_results = [
                executor.submit(extract_task, doc=doc, search_term=description)
                for doc in docs
            ]

            # Collect the results as they complete
            extracted_docs = [
                future.result()
                for future in concurrent.futures.as_completed(future_results)
            ]
        # Get summary
        summaries = ""
        sumamry_id_list = []
        summary_list = []
        summary_tokens = 0
        summary_tokens_list = []
        for doc_id, content in extracted_docs:
            if """DOCUMENT NOT RELATED""" not in content:
                summary_list.append(f"Content: {content}\nSource: {doc_id}")
                sumamry_id_list.append(doc_id)
                summary_tokens += num_tokens_from_string(content)
                summary_tokens_list.append(num_tokens_from_string(content))

        target_tokens = 2048
        if summary_tokens > target_tokens:
            # truncate
            target_tokens = int(target_tokens / len(summary_tokens_list))
            summary_list = [
                split_by_token(string=summary, chunk_length=target_tokens)[0]
                for summary in summary_list
            ]

        summaries = "\n\n".join(summary_list)

        summaries = "NO INFO FOUND" if summaries == "" else summaries

        logger.info(f"[extract_and_combine.run] - SUMMARY: {summaries}")
        logger.info(
            f"[extract_and_combine.run] - SUMMARY Length: {num_tokens_from_string(summaries)}"
        )
        logger.info(
            f"[extract_and_combine.run] - SUMMARY Length of each: {summary_tokens_list}"
        )

        # Generate response json with the knowledge and references
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            template=SYSTEM_PROMPT
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            HUMAN_PROMPT, output_parser=self.output_parser
        )

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        response = self.llm(
            chat_prompt.format_prompt(
                question=description,
                research_summary=summaries,
                format_instructions=self.format_instructions,
            ).to_messages()
        )

        response_json = self.output_parser.parse(response.content)

        # Formatting references and removing duplicates just in case
        logger.info(f"response_json: {response_json}")
        used_ids = set()
        # TODO: refactor as one
        used_docs = [
            doc
            for doc in docs
            if (
                doc.metadata["source"] in response_json["references"]
            )  # If it was used by the LLM
            or (
                doc.metadata["source"] in docs_by_id
            )  # If the engineer added it to the list
            or (
                doc.metadata["source"] in sumamry_id_list
            )  # if it was extracted as related
        ]
        used_docs = [
            doc
            for doc in used_docs
            if (doc.metadata["source"], doc.metadata["id"])
            not in used_ids  # source is the id
            and not used_ids.add(
                (doc.metadata["source"], doc.metadata["id"])
            )  # will always be true so won't affect the condition
        ]
        logger.info(f"used_docs: {used_docs}")

        references = []
        references2show = []  # The ones that can be displayed in an email

        for tool in tools:
            tool_docs = [
                doc for doc in used_docs if doc.metadata["pulled_by"] == tool.name
            ]
            tool_references = [
                tool.format_reference(
                    {
                        "id": doc.metadata["source"],
                        "title": doc.metadata["title"],
                        "url": doc.metadata.get("url", ""),
                        "document": doc.metadata.get("document", ""),
                        "visibility": doc.metadata.get("visibility", ""),
                    }
                )
                for doc in tool_docs
            ]
            references.extend(tool_references)
            logger.debug(f"reference from {tool.name}: {tool_references}")

            tool_references = [
                tool.format_reference(
                    {
                        "id": doc.metadata["source"],
                        "title": doc.metadata["title"],
                        "url": doc.metadata["url"],
                        "document": doc.metadata.get("document", ""),
                        "visibility": doc.metadata.get("visibility", ""),
                    }
                )
                for doc in tool_docs
                if (tool.is_displayable)
                and (doc.metadata["visibility"] in ["public"])
                and ("url" in doc.metadata)
            ]
            references2show.extend(tool_references)
            logger.debug(
                f"publicly shown reference from {tool.name}: {tool_references}"
            )

        response_json["references"] = references
        response_json["public_html_references"] = ""

        # Include references within the HTML email content
        if len(references2show) > 0:
            reference_appendix = "<strong>References:</strong><br><ul>"
            for reference in references2show:

                if "url" in reference:
                    reference_appendix += "<li>" + reference["source"] + ": "
                    print("[extract_and_combine] - URL present in doc!")
                    reference_appendix += (
                        '<a href="'
                        + reference["url"]
                        + '">'
                        + reference["title"]
                        + "</a>"
                    )
                    reference_appendix += "</li>"
            response_json["public_html_references"] = reference_appendix + "</ul>"
            logger.debug(
                f"public_html_references: {response_json['public_html_references']}"
            )

        return response_json
