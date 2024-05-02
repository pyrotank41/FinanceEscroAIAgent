import os
from rag import LlamaIndexRag
from llama_index.core.chat_engine import ContextChatEngine
from typing import List, Optional
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor

from utility.utils import get_openai_api_key, load_prompt
from loguru import logger

PROMPT = "you are an ai assistant and your name is jeff"

RAG_CONTEXT_TEMPLATE= (
    "# Knowledge Context"
    "{context_str}"
)


class EscrowAssistant():

    def __init__(self, 
                 llm=None,
                 llm_for_rag=None,
                 system_prompt=None,
                 chat_history:Optional[List[ChatMessage]]=None,
                 prompt_file:str = "prompts/prompt_3_for_phi.txt"):
        
        if system_prompt is None:
            system_prompt = load_prompt(prompt_file)
        
        if not llm:
            self.llm = OpenAI(model="gpt-3.5-turbo",
                              api_key=get_openai_api_key(),
                              temperature=0.1)
            logger.info("No LLM provided, defaulting to OpenAI LLM")
        else:
            self.llm = llm
        
        rag = LlamaIndexRag(llm=llm_for_rag)
        if system_prompt is None:
            system_prompt = PROMPT
            
        # Based on our experiments, we found that sentence window retrieval
        # with size 1 works best for our use case
        self.index = rag.get_sentence_window_index(window_size=1)
        logger.info("Index created successfully")
        
        # define postprocessors
        postproc = MetadataReplacementPostProcessor(
            target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=2, model="BAAI/bge-reranker-base"
        )

        chat_engine = ContextChatEngine.from_defaults(
            llm=self.llm,
            retriever=self.index.as_retriever(
                    similarity_top_k=6,
                ),
            prefix_messages=[
                ChatMessage(
                    role="system",
                    content=system_prompt
                )
            ],
            context_template=RAG_CONTEXT_TEMPLATE,
            postprocessors=[postproc, rerank]

        )
        
        if chat_history is not None:
            chat_engine._memory.set(chat_history)
    
        self.chat_engine = chat_engine

    def streaming_chat_repl(self) -> None:
        """Enter interactive chat REPL with streaming responses."""
        print("===== Entering Chat REPL =====")
        print('Type "exit" to exit.\n')
        self.chat_engine.reset()
        message = input("User: ")
        while message != "exit":
            response = self.chat_engine.stream_chat(message)
            print("Assistant: ", end="", flush=True)
            response.print_response_stream()
            print("\n")
            message = input("User: ")


if __name__ == "__main__":
    
    
    assistant = EscrowAssistant(system_prompt=PROMPT)
    assistant.streaming_chat_repl()
    
#     print(agent)
#     print(agent("hello, what's your name?"))
