from llama_index.core.chat_engine.types import (
    StreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.callbacks import trace_method
from threading import Thread
import os
from rag import LlamaIndexRag
from llama_index.core.chat_engine import ContextChatEngine
from typing import List, Optional
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor

from utility.utils import get_openai_api_key, load_prompt
from loguru import logger


RAG_CONTEXT_TEMPLATE= (
    "# Knowledge Context"
    "{context_str}"
)


class CustomContextChatEngine(ContextChatEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @trace_method("chat")
    def stream_chat( # overriding the stream_chat method to add context relevance check
        self, message: str, chat_history: Optional[List[ChatMessage]] = None, context_relevance_threshold: float = 0.5
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))
        context_str_template, nodes = self._generate_context(message)

        ### Context Relevance Check
        message_for_validataion = [
            ChatMessage(
                role=self._llm.metadata.system_role,
                content=" following is the text from my database\n" + context_str_template +
               "does the text have an answer to the query? respond only between: 0.0 (not relevant at all) to 1.0 (has relevant information)"
            ),
            ChatMessage(
                role="user",
                content=message
            )
        ]

        logger.debug(context_str_template)
        relevance = self._llm
        relevance = self._llm.chat(message_for_validataion)

        # Function to check and extract float
        def extract_floats(arr):
            floats = []
            for item in arr:
                try:
                    # Attempt to convert each item to a float
                    floats.append(float(item))
                except ValueError:
                    # If conversion fails, pass
                    continue
            return floats

        # Extracting floats
        float_numbers = extract_floats(relevance.__str__().split())

        if len(float_numbers) > 0:
            relevance = float_numbers[0]
            logger.info(f"relevance:{relevance}")
            try:
                if relevance < context_relevance_threshold:
                    logger.info(
                        "Context Relevance is less then the threshold! resetting content of template to 'No relevant information found'")
                    context_str_template = self._context_template.format(
                        context_str="No relevant information found")
            except ValueError:
                pass
        ### Context relevance check ends here

        prefix_messages = self._get_prefix_messages_with_context(
            context_str_template)
        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(all_messages),
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
        thread = Thread(
            target=chat_response.write_response_to_history, args=(
                self._memory,)
        )
        thread.start()

        return chat_response


class EscrowAssistant():

    def __init__(self, 
                 llm=None,
                 llm_for_rag=None,
                 system_prompt=None,
                 chat_history:Optional[List[ChatMessage]]=None,
                 prompt_file:str = "prompts/prompt_3.txt"):

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

        chat_engine = CustomContextChatEngine.from_defaults(
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
    assistant = EscrowAssistant(system_prompt=load_prompt("prompts/prompt_main.txt"))
    assistant.streaming_chat_repl()
    
#     print(agent)
#     print(agent("hello, what's your name?"))
