import os
BACKEND_URL = os.environ.get("BACKEND_URL") or "http://localhost:8000"

import streamlit as st
import asyncio
import websockets
import uuid
import requests
import json
# Function to handle WebSocket communication with FastAPI
async def send_message(session_id, message):
    uri = f"ws://{BACKEND_URL}/ws/{session_id}"  # Adjust the URL if needed
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(message)  # Send the user message
            response = await websocket.recv()  # Wait for the response
            return response
    except Exception as e:
        st.error(f"Error with WebSocket connection: {e}")
        return None

# Run asyncio loop in a separate thread to be compatible with Streamlit
def run_asyncio_loop(session_id, message):
    return asyncio.run(send_message(session_id, message))

# Streamlit app UI
st.title("Welcome to Tutor Helper! 👋")

# Generate a new session_id (UUID) on app load and store it in session state
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())  # Automatically generate a new session ID

session_id = st.session_state['session_id']  # Use the stored session ID

# # Display the session ID (optional, for debugging purposes)
# st.sidebar.write(f"Session ID: {session_id}")

# Create chat interface
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Greetings! I'm your assistant.\n Together we can craft any tutor classes you want."}
    ]

# Display previous chat history
for message in st.session_state['messages']:
    st.chat_message(message['role']).markdown(message['content'])

# Handle user input
user_message = st.chat_input("Type your message...")

if user_message:
    # Add the user message to the chat history
    st.session_state['messages'].append({"role": "user", "content": user_message})
    # Display the user message
    st.chat_message("user").markdown(user_message)
    
    # Send the user message to the WebSocket backend and get the response
    bot_response = run_asyncio_loop(session_id, user_message)
    bot_response = json.loads(bot_response)
    # Add the bot response to the chat history
    st.session_state['messages'].append({"role": "assistant", "content": bot_response["content"]})
    
    # Display the latest chat messages from asstistant
    st.chat_message("assistant").markdown(bot_response["content"])


# Function to search for tutor terms via FastAPI
def search_terms(query: str):
    try:
        response = requests.post(f"http://{BACKEND_URL}/search/ts", json={"query_search": query})
        if response.status_code == 200:
            return response.json()  # Assuming the response is in JSON format
        else:
            st.error(f"Search failed with status code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error in search request: {e}")
        return None

# Search bar in the sidebar
search_query = st.sidebar.text_input(label="Search", placeholder="what is singleton?")

# Show search results in the sidebar if a query is provided
if search_query:
    results = search_terms(search_query)
    if results:
        st.sidebar.write(f"### Search Results for *{search_query}*:")
        for result in results:
            # Extract necessary fields from result
            link = result.get('url', '#')  # Ensure link is part of the result
            title = result.get('title', 'No Title')
            description = result.get('description', 'No Description')
            with st.sidebar.container():
                st.sidebar.markdown(
            f"""
            <div style='border: 1px solid #ddd; padding: 10px; margin: 5px 0;'>
                <a href="{link}" target="_blank" style='text-decoration: none; color: inherit;'>
                    <h4>{title}</h4>
                    <p>{description[:100]+"..."}</p>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.write("No results found.")