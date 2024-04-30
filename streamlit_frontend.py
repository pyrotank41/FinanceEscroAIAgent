from llama_index.core.memory import ChatMemoryBuffer
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)

from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings import resolve_embed_model
import streamlit as st
from utility.utils import get_openai_api_key

# Setup
llm = OpenAI(
    api_key=get_openai_api_key(),
    model="gpt-3.5-turbo",
    temperature=0.1
)
Settings.llm = llm
# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# Load the docs
# check if storage already exists
PERSIST_DIR = "./storage/escrow_data/simple_rag"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("escrow_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

memory = ChatMemoryBuffer.from_defaults(token_limit=8500)
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    llm=llm,
    system_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an escrow 1024.17 Document."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=True,
)


st.markdown("# Escro AI Agent")
st.markdown("*(1024.17 Document Chat)*")


def query_response(query):
    response = chat_engine.stream_chat(query)
    return response


# Initilize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app reun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input("Message Escro Agent..."):
    st.chat_message('user').markdown(prompt)

    with st.chat_message("assistant"):
        response = query_response(prompt)
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {'role': 'user', 'content': prompt})

    st.session_state.messages.append(
        {'role': 'assistant', 'content': full_response})
