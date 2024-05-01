import time
from rag import LlamaIndexRag
from llama_index.core.chat_engine import ContextChatEngine
from typing import Any, List, Optional
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
    ToolOutput,
)

from utility.utils import get_openai_api_key

prompt = "you are an ai assistant and your name is jeff"




# response = chat_engine.chat("What is an escrow account?")

class EscrowAssistant():

    def __init__(self, llm=None,
                 system_prompt=None,
                 chat_history: Optional[List[ChatMessage]]=None):
        
        if not llm:
            self.llm = OpenAI(model="gpt-3.5-turbo",
                              api_key=get_openai_api_key(),
                              temperature=0.1)
        else:
            self.llm = llm
        
        rag = LlamaIndexRag()
        
        # Based on our experiments, we found that sentence window retrieval
        # with size 1 works best for our use case
        self.index = rag.get_sentence_window_index(window_size=1)
        

        chat_engine = ContextChatEngine.from_defaults(
            llm=self.llm,
            retriever=self.index.as_retriever(),
            prefix_messages=[
                ChatMessage(
                    role="system",
                    content=system_prompt
                )
            ]
        )
        
        if chat_history is not None:
            chat_engine._memory.set(chat_history)
    
        self.chat_engine = chat_engine

    def __call__(self, 
                 message:str
                 ) -> AgentChatResponse:
        
        self.chat_engine._memory.put(ChatMessage(content=message, role="user"))

        context_str_template, nodes = self.chat_engine._generate_context(message)
        prefix_messages = self.chat_engine._get_prefix_messages_with_context(
            context_str_template)
        prefix_messages_token_count = len(
            self.chat_engine._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self.chat_engine._memory.get(
            initial_token_count=prefix_messages_token_count
        )
        chat_response = self.chat_engine._llm.chat(all_messages)
        ai_message = chat_response.message
        self.chat_engine._memory.put(ai_message)

        return AgentChatResponse(
            response=str(chat_response.message.content),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )

        # return self.assistant.stream_chat(query)


if __name__ == "__main__":
    
    
    assistant = EscrowAssistant(system_prompt=prompt)
    assistant.chat_engine.streaming_chat_repl()
    
#     print(agent)
#     print(agent("hello, what's your name?"))
