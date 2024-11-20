from fastapi.testclient import TestClient
from tutor_helper.use_cases.fastapi import app
import pytest
import logging
import os

logger = logging.getLogger("test_logger")

# This is your test
class TestGeneralAPI:
    @classmethod
    def setup_class(cls):
        cls.client = TestClient(app)

class TestSearchAPI:
    @classmethod
    def setup_class(cls):
        cls.client = TestClient(app)

    def test_simple_search(self):
        query_search = "Retro Scan registration unsuccessful error SEG-43837"
        response = self.client.post(
            "/search/ts",
            json={
                "query_search": query_search,
                "language": "english",
                "scope_meta": query_search,
            },
        )
        assert response.status_code == 200
        tools = set([i["tool"] for i in response.json()])
        expected_tools = {
            "DuckDuckGoSearch"
        }

        missing_tools = expected_tools - tools
        extra_tools = tools - expected_tools
        assert (
            tools == expected_tools
        ), f"The sets are not equal. Missing: {missing_tools}. Extra: {extra_tools}"

    def test_search_term(self):
        description = "Retro Scan registration unsuccessful error SEG-43837"
        response = self.client.post(
            "/llm/actions/create_search_term",
            json={"description": description},
        )
        assert response.status_code == 200
        assert response.json() != description

if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
