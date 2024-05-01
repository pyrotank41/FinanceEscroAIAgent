import os
from typing import Any, List
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
    Document
)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes, SentenceSplitter
from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.evaluation import QueryResponseEvaluator, ContextRelevancyEvaluator

from pydantic import BaseModel
from trulens_eval import (
    Tru,
    Feedback,
    TruLlama,
    OpenAI as trulens_oai
)
from trulens_eval.feedback import Groundedness
import numpy as np
from utility.utils import get_openai_api_key

import nest_asyncio

nest_asyncio.apply()



class RAGengine(BaseModel):
    """
    Represents a RAG (Retriever, Answer Generator) engine.

    Attributes:
        engine (RetrieverQueryEngine): The retriever query engine used by the RAG engine.
        name (str): The name of the RAG engine.
    """
    engine: Any
    name: str


class RAGeval():
    """ 
    Rag evaluation

    In an AI app, there are three main parameter as follows:
    Query (eg user message to the system)
    Response (message generated from system)
    Context (Information provided to system to generate response)

    We evaluate the following to optimize our RAG app (Domain = [0,1]):
    Answer Relevance (Response -> Query): Response relevant to the query?
    Context Relevance ( Query -> Context): Is the response supported by the context?
    Groundedness (context -> Response): Retrieved context relevant to the query?

    for more information, check out the following link:
    https://www.trulens.org/trulens_eval/getting_started/core_concepts/rag_triad/

    """

    def __init__(self):
        self.tru = Tru()

    def get_answer_relevance_feedback(self, provider):
        # Answer Relevance (Response -> Query): Response relevant to the query?
        return (
            Feedback(
                # (chain of thought reasoning, to get the reasoning behing the score)
                provider.relevance_with_cot_reasons,
                name="Answer Relevance"
            )
            .on_input_output()
        )

    def get_context_relevance_feedback(self, provider, context_selected):

        # Context Relevance ( Query -> Context): Is the response supported by the context?
        return (
            Feedback(
                provider.relevance_with_cot_reasons,
                name="Context Relevance")
            .on_input()  # pointer to user query
            .on(context_selected)
            # aggrigate score for all the retrieved context
            .aggregate(np.mean)
        )

    def get_groundness_feedback(self, provider, context_selected):

        # Groundedness (context -> Response): Retrieved context relevant to the query?
        grounded = Groundedness(groundedness_provider=provider)
        return (
            Feedback(grounded.groundedness_measure_with_cot_reasons,
                     name="Groundedness")
            .on(context_selected)
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
        )

    def get_trulens_recorder(self,
                             eval_engine: RetrieverQueryEngine,
                             eval_id: str,
                             ):
        # llm to run feedback on; we are using openai, defaults to gpt 3.5
        provider = trulens_oai()

        # Pointer to retrieved context (used by trulens for retreving intermidiate results)
        context_selected = TruLlama.select_source_nodes().node.text

        # as shared before, we will use the following feedbacks in our evaluation
        # note, based on the application, we can add more feedbacks
        answer_relevance = self.get_answer_relevance_feedback(provider)
        context_relevance = self.get_context_relevance_feedback(
            provider, context_selected)
        groundedness = self.get_groundness_feedback(provider, context_selected)

        feedbacks = [answer_relevance, context_relevance, groundedness]

        # as the name suggests, this is the trulens recorder,
        # it records the during the evaluation cycle to evaluate our RAG methods
        tru_recorder = TruLlama(
            eval_engine,
            app_id=eval_id,
            feedbacks=feedbacks
        )

        return tru_recorder

    def get_queries(self, eval_queries_path):
        eval_queries = []
        print(eval_queries_path)
        with open(eval_queries_path, 'r') as file:
            for line in file:
                # Remove newline character and convert to integer
                item = line.strip()
                eval_queries.append(item)
        return eval_queries

    def evaluate_rag(self,
                     eval_engines: List[RAGengine],
                     eval_queries_path: str,
                     reset_database: bool = False):

        if reset_database:
            self.tru.reset_database()
        eval_queries = self.get_queries(eval_queries_path)

        # Evaluate each engine
        for eval_engine in eval_engines:

            # Get the trulens recorder
            tru_recorder = self.get_trulens_recorder(
                eval_engine.engine, eval_id=eval_engine.name)

            # Evaluate the engine on each query
            for question in eval_queries:
                with tru_recorder as recording:
                    eval_engine.engine.query(question)

class LlamaIndexRag():
    def __init__(
        self,
        llm=None,
        local_index_storage_dir="./rag_index_storage/",
        data_directory="escrow_data",
        embedding_model="local:BAAI/bge-small-en-v1.5"
    ):

        if llm is None:
            self.llm = OpenAI(
                api_key=get_openai_api_key(),
                model="gpt-3.5-turbo",
                temperature=0.1
            )
        else:
            self.llm = llm

        # Setup
        Settings.llm = self.llm
        Settings.embed_model = resolve_embed_model(embedding_model)

        self.local_index_storage_dir = local_index_storage_dir
        self.data_directory = data_directory
        self.document = None

    def _get_stored_index(self, persist_dir):
        # load the existing index
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        return index

    def _get_persist_dir_path(self, type_name="basic_rag"):
        return os.path.join(self.local_index_storage_dir,
                            self.data_directory, type_name)

    def _get_documents(self):
        documents = SimpleDirectoryReader(self.data_directory).load_data()
        document = Document(text="\n\n".join([doc.text for doc in documents]))
        return [document]

    def get_direct_rag_index(self,
                             chunk_sizes=None,
                             chunk_overlap=None,
                             use_persisted=True):

        if chunk_sizes is not None:
            Settings.chunk_size = chunk_sizes

        if chunk_overlap is not None:
            Settings.chunk_overlap = chunk_overlap

        persist_dir = self._get_persist_dir_path()

        if os.path.exists(persist_dir) and use_persisted:
            index = self._get_stored_index(persist_dir)
        else:
            index = VectorStoreIndex.from_documents(self._get_documents())
            index.storage_context.persist(persist_dir=persist_dir)

        return index

    def get_sentence_window_index(self,
                                  window_size=3,
                                  chunk_sizes=128,
                                  chunk_overlap=20,
                                  use_persisted=True
                                  ):

        text_splitter = SentenceSplitter(
            chunk_size=chunk_sizes,
            chunk_overlap=chunk_overlap
        )
        Settings.text_splitter = text_splitter

        persist_dir = self._get_persist_dir_path("sentence_window")
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        Settings.node_parser = node_parser

        if os.path.exists(persist_dir) and use_persisted:
            index = self._get_stored_index(persist_dir)

        else:
            index = VectorStoreIndex.from_documents(self._get_documents())
            index.storage_context.persist(persist_dir=persist_dir)

        return index

    def get_automerging_index(
        self,
        chunk_sizes=None,
    ):
        persist_dir = self._get_persist_dir_path("automerging_index")
        chunk_sizes = chunk_sizes or [1024, 256]  # [2048, 512, 128]
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes)
        nodes = node_parser.get_nodes_from_documents(self._get_documents())
        leaf_nodes = get_leaf_nodes(nodes)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        if not os.path.exists(persist_dir):
            index = VectorStoreIndex(
                leaf_nodes, storage_context=storage_context)
            index.storage_context.persist(persist_dir=persist_dir)
        else:
            index = self._get_stored_index(persist_dir)
        return index

    def generate_direct_rag_engine(self,
                                   similarity_top_k=4,
                                   chunk_sizes=None,
                                   chunk_overlap=None,
                                   use_persisted=True):
        index = self.get_direct_rag_index(
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap,
            use_persisted=use_persisted
        )

        direct_rag_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k)

        return direct_rag_engine

    def generate_sentence_window_engine(
        self,
        similarity_top_k=6,
        rerank_top_n=2,
        chunk_sizes=128,
        window_size=3,
        chunk_overlap=20,
        rerank_model="BAAI/bge-reranker-base",
        use_persisted=True
    ):

        index = self.get_sentence_window_index(
            window_size=window_size,
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap,
            use_persisted=use_persisted)

        # define postprocessors
        postproc = MetadataReplacementPostProcessor(
            target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model=rerank_model
        )

        sentence_window_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            node_postprocessors=[postproc, rerank]
        )
        return sentence_window_engine

    def generate_automerging_engine(
        self,
        similarity_top_k=12,
        rerank_top_n=6,
        chunk_sizes=None,
        rerank_model="BAAI/bge-reranker-base"
    ):
        index = self.get_automerging_index(chunk_sizes=chunk_sizes)

        base_retriever = index.as_retriever(
            similarity_top_k=similarity_top_k)

        retriever = AutoMergingRetriever(
            base_retriever, index.storage_context, verbose=True
        )
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model=rerank_model
        )
        auto_merging_engine = RetrieverQueryEngine.from_args(
            retriever, node_postprocessors=[rerank]
        )
        return auto_merging_engine


def evaluate_rag(
        eval_queries_path: str = './application_llm_eval/valid_eval_queries_rag.txt',
        reset_database: bool = False
):
    llama_rag = LlamaIndexRag()
    rag_evaluator = RAGeval()

    # direct rag experiments to find the best chunnk size for the index for our application
    direct_rag_engine_64_chunk = llama_rag.generate_direct_rag_engine(chunk_sizes=64, use_persisted=False)
    direct_rag_engine_64_chunk_overlap_10 = llama_rag.generate_direct_rag_engine(chunk_sizes=64, chunk_overlap=10, use_persisted=False)
    direct_rag_engine_128_chunk = llama_rag.generate_direct_rag_engine(chunk_sizes=128, use_persisted=False) 
    direct_rag_engine_256_chunk = llama_rag.generate_direct_rag_engine(
        chunk_sizes=256, use_persisted=False)  # generated the best evaluation for direct rag
    direct_rag_engine_512_chunk = llama_rag.generate_direct_rag_engine(chunk_sizes=512, use_persisted=False)
    direct_rag_engine_1024_chunk = llama_rag.generate_direct_rag_engine(chunk_sizes=1024, use_persisted=False)

    # # sentence window experiments
    sentence_window_engine_1_window = llama_rag.generate_sentence_window_engine(window_size= 1, use_persisted=False) # Generated best evaluation overall
    sentence_window_engine_2_window = llama_rag.generate_sentence_window_engine(window_size= 2, use_persisted=False)
    sentence_window_engine_3_window = llama_rag.generate_sentence_window_engine(window_size= 3, use_persisted=False) 
    sentence_window_engine_4_window = llama_rag.generate_sentence_window_engine(window_size=4, use_persisted=False)

    # automerging experiments
    automerging_engine_3_step_64 = llama_rag.generate_automerging_engine(chunk_sizes=[1024,256,64])
    automerging_engine_3_step_128 = llama_rag.generate_automerging_engine(chunk_sizes=[2048,513,128])
    automerging_engine_2_step_64 = llama_rag.generate_automerging_engine(chunk_sizes=[1024,64])
    automerging_engine_2_step_128 = llama_rag.generate_automerging_engine(chunk_sizes=[2048,128])

    eval_engines = [
        RAGengine(engine=direct_rag_engine_1024_chunk, name="direct_rag 1024 chunk"),
        RAGengine(engine=direct_rag_engine_512_chunk, name="direct_rag 512 chunk"),
        RAGengine(engine=direct_rag_engine_256_chunk, name="direct_rag 256 chunk"),
        RAGengine(engine=direct_rag_engine_128_chunk, name="direct_rag 128 chunk"),
        RAGengine(engine=direct_rag_engine_64_chunk, name="direct_rag 64 chunk"),
        RAGengine(engine=direct_rag_engine_64_chunk_overlap_10, name="direct_rag 64 chunk overlap 10"),
        RAGengine(engine=sentence_window_engine_1_window,
                  name="sentence_window: 1 window"),
        RAGengine(engine=sentence_window_engine_2_window,
                  name="sentence_window: 2 window"),
        RAGengine(engine=sentence_window_engine_3_window,
                  name="sentence_window: 3 window"),
        RAGengine(engine=sentence_window_engine_4_window,
                  name="sentence_window: 4 window"),
        RAGengine(engine=automerging_engine_2_step_64,
                  name="automerging 2 step 64"),
        RAGengine(engine=automerging_engine_2_step_128,
                  name="automerging 2 step 128"),
        RAGengine(engine=automerging_engine_3_step_64,
                  name="automerging 3 step 64"),
        RAGengine(engine=automerging_engine_3_step_128,
                  name="automerging 3 step 128"),

    ]

    rag_evaluator.evaluate_rag(
        eval_engines, eval_queries_path, reset_database)


if __name__ == "__main__":

    # Run the evaluations
    llama_rag = LlamaIndexRag()
    evaluate_rag(reset_database=False) # set to True to reset the evaluation database

    # Save the evaluations to a csv file
    # highlevel_eval = Tru().get_leaderboard(app_ids=[])
    # print(highlevel_eval)
    # # save the evaluation results
    # highlevel_eval.to_csv("eval_results/rag_eval_results.csv")

    # Run the dashboard to inspect the evaluations in detail
    # Tru().run_dashboard()
