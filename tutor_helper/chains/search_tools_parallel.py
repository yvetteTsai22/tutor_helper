from tutor_helper.tools.utilities.utils import num_tokens_from_string, split_by_token
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Any, Dict, List, Optional
import concurrent.futures
import unicodedata
import re



class SearchToolsParallel:
    def __init__(self, tools: List):
        self.tools = tools

    def run(self, search_term: str, product: str, docs_by_id: List = []):
        raw_docs = []

        # Create tasks for each tool and gather the results
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the tasks to the executor
            future_results = [
                executor.submit(
                    self.execute_tool_with_trace, tool, search_term, product, docs_by_id
                )
                for tool in self.tools
            ]

            # Collect the results as they complete
            response_docs = [
                future.result()
                for future in concurrent.futures.as_completed(future_results)
            ]

        # Flatten the results
        for docs in response_docs:
            raw_docs.extend(docs)

        return raw_docs

    def execute_tool_with_trace(self, tool, search_term, product, docs_by_id):
        response = tool.from_description(search_term)

        if len(docs_by_id) > 0:
            from_notes = tool.from_ids(docs_by_id)

            response.extend(from_notes)


        return response

    def chunk(self, docs: List, token_length: int = 3072):
        # Splitting documents which content is > than XXXX tokens
        # Create multiple documents with the same metadata and splitted content
        # This is to avoid the 2048 tokens limit of the LLM

        new_docs = []
        for doc in docs:
            content_length = num_tokens_from_string(doc["content"])
            if content_length > token_length:
                # Split the content into chunks
                # TODO: remove the fallback if split_by_token is stable
                try:
                    splitted_content = split_by_token(
                        string=doc["content"], chunk_length=token_length
                    )
                except Exception as e:
                    print(f"Error while calling split_by_token(): {e}")
                    content_words = doc["content"].split()
                    splitted_content = [
                        " ".join(content_words[i : i + token_length])
                        for i in range(0, len(content_words), token_length)
                    ]

                # Create new documents with updated content
                for content in splitted_content:
                    new_doc = doc.copy()
                    new_doc["content"] = content
                    new_docs.append(new_doc)
            else:
                # No splitting required, add the original document
                new_docs.append(doc)

        return new_docs

    def transform(self, docs: List):

        transformed_docs = []
        for doc in docs:
            content = unicodedata.normalize("NFKD", doc["content"])
            content = re.sub(r"\s{2,}", " ", content)
            content = re.sub(
                r"\(https:\/\/powerbox-na-file.trend.org\/SFDC\/DownloadFile_iv\.php.+?\)",
                "",
                content,
            )

            doc_object = Document(
                page_content="Title: "
                + doc["title"]
                + "\n"
                + content,
                metadata={k: v for k, v in doc.items() if k != "content"},
            )

            transformed_docs.append(doc_object)

        return transformed_docs
