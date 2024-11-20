from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from tutor_helper.schema.payload import (
    SearchPayload,
    SearchTermPayload,
)
from tutor_helper.tools.search.parallel_search import ParallelSearch
from tutor_helper.tools.search.search_term import SearchTerm


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
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        logger.info("Client disconnected")