from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from tutor_helper.schema.payload import (
    SearchPayload,
    SearchTermPayload,
)
from tutor_helper.tools.search.parallel_search import ParallelSearch
from tutor_helper.tools.search.search_term import SearchTerm

from tutor_helper.agents.tutor_assistant.base import (
    chat_agent,
)
from tutor_helper.agents.tutor_assistant.toolkit import SimplifiedToolkit
from tutor_helper.common.llms import LlmLoader
from tutor_helper.prompts.templates.chat_agent import ChatResponseWithKB
from tutor_helper.output_parsers.agent_parser import NewAgentOutputFixingParser


import json
import logging
logger = logging.getLogger(__name__)

app = FastAPI()


# @async_trace_decorator
@app.post("/search/ts")
async def search_ts(payload: SearchPayload):
    results = ParallelSearch()(payload)
    return results


# @async_trace_decorator
@app.post("/llm/actions/create_search_term")
async def create_search_term_action(payload: SearchTermPayload):
    return SearchTerm().from_description(payload.description)


# Create a /health endpoint that returns 200 OK
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Client sent: {data}")
            # prepare tools
            toolkit = SimplifiedToolkit()

            # prepare llm
            chat_model = LlmLoader.create_chat_llm(
                model=LlmLoader.DEPLOYMENT_35_TURBO_LG, temperature=0.5, top_p=0.5
            )
            agent_kwargs = {
                "prefix": ChatResponseWithKB.SYSTEM_MESSAGE_WITH_TOOLS_PREFIX,
                "format_instructions": ChatResponseWithKB.FORMAT_INSTRUCTIONS_FOR_AGENT,
                "output_parser": NewAgentOutputFixingParser(),
                "input_variables": ChatResponseWithKB.INPUT_VARIABLES,
                "return_intermediate_steps": True,
            }
            # Initialize the agent with the tools and language model
            agent = chat_agent(
                llm=chat_model, 
                toolkit=toolkit, 
                verbose=True, 
                agent_kwargs=agent_kwargs
            )
            response = agent.run(input={"similarity_search_term": data, "request_raw_question_input": data})
            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.info("Client disconnected")