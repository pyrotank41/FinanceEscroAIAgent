from llama_index.core.memory import ChatMemoryBuffer
from loguru import logger

from llama_index.llms.openai import OpenAI
import streamlit as st
from chat_assistant import EscrowAssistant
from utility.utils import get_openai_api_key

from llama_index.llms.ollama import Ollama


st.markdown("# Escro Chat Assistant")
st.markdown("*(1024.17 Document Chat)*")

# Initilize chat history
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# display chat messages from history on app reun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if "assistant" not in st.session_state.keys():  # Initialize the chat engine
    assistant = EscrowAssistant()
    st.session_state.assistant = assistant


# React to user input
if prompt := st.chat_input("Message Escro Agent..."):
    st.chat_message('user').markdown(prompt)

    with st.chat_message("assistant"):
        response = st.session_state.assistant.chat_engine.stream_chat(
            prompt)
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)
        # logger.info(
        #     st.session_state.assistant.chat_engine._memory.get())

    st.session_state.messages.append(
        {'role': 'user', 'content': prompt})

    st.session_state.messages.append(
        {'role': 'assistant', 'content': full_response})