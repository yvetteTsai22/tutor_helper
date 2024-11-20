from langchain_community.tools import DuckDuckGoSearchResults
from tutor_helper.tools.contracts.document_picker import DocumentPickerTool
from typing import List, Dict, Optional
from langchain.schema import Document
import json

class DuckDuckGoSearch(DocumentPickerTool):
    DocumentPickerTool.initialize_formats()

    id_key:str = "url"
    display_name:str = "DuckDuckGoSearch"
    name:str = "DuckDuckGoSearch"
    description:str = (
        "Performs searches using the DuckDuckGo search engine and retrieves relevant results."
        f"\nInput Format: {DocumentPickerTool.input_format}. "
        f"\nOutput format: {DocumentPickerTool.output_format}"
    )
    num_results:int = 8
    # to determine whether to display the result
    is_displayable:bool = True
    index_fields:dict = {
        "title": "title",
        "description":"snippet",
        "url":"url"
    }

    def _get_matching_docs(
        self, search_term: str, num_results: int=8, **kwargs
    ) -> List[Document]:
        
        search = DuckDuckGoSearchResults(output_format="list", num_results=num_results)

        search_results = search.invoke(search_term)
        return [
            Document(
            page_content=doc["snippet"],
            metadata={"title": doc["title"], "url": doc["link"], "snippet": doc["snippet"]}
            )
            for doc in search_results
        ]

    def _transform_docs(self, docs: List) -> List[Dict]:
        return [
            {   
                "tool": self.name,
                "display_name": self.display_name,
                "title": doc.metadata[self.index_fields["title"]],
                "description": doc.page_content,
                "url": doc.metadata[self.index_fields["url"]],
            }
            for doc in docs
        ]        
    
    def _run(self, description: str, product: Optional[str] = None) -> str:
        return super()._run(
            json.dumps({"description": description, "product": product})
        )