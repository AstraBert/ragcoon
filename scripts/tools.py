from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform, StepDecomposeQueryTransform
from llama_index.core.query_engine import TransformQueryEngine, MultiStepQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from os import environ as ENV
import json
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

embed_model = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-modernbert-base")
node_parser = SemanticSplitterNodeParser(embed_model=embed_model)

qc = QdrantClient("http://localhost:6333")
aqc = AsyncQdrantClient("http://localhost:6333")

if not qc.collection_exists("data"):
    docs = SimpleDirectoryReader(input_dir="../data/").load_data()
    nodes = node_parser.get_nodes_from_documents(docs, show_progress=True)
    vector_store = QdrantVectorStore("data", qc, aclient=aqc, enable_hybrid=True, fastembed_sparse_model="Qdrant/bm25")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model, storage_context=storage_context)
    index1 = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    index2 = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
else:
    vector_store = QdrantVectorStore("data", client=qc, aclient=aqc, enable_hybrid=True, fastembed_sparse_model="Qdrant/bm25")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    index1 = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    index2 = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

class EvaluateContext(BaseModel):
    context_is_ok: int = Field(description="Is the context relevant to the question? Give a score between 0 and 100")
    reasons: str = Field(description="Explanations for the given evaluation")

llm = Groq(model="llama-3.3-70b-versatile", api_key=ENV["GROQ_API_KEY"])
llm_eval = llm.as_structured_llm(EvaluateContext)
Settings.llm = llm
faith_eval = FaithfulnessEvaluator()
rel_eval = RelevancyEvaluator()

query_engine = index.as_query_engine(llm=llm)
query_engine1 = index1.as_query_engine(llm=llm)
query_engine2 = index2.as_query_engine(llm=llm)

hyde = HyDEQueryTransform(llm=llm, include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine=query_engine1, query_transform=hyde)

step_decompose_transform = StepDecomposeQueryTransform(llm, verbose=True)
multistep_query_engine = MultiStepQueryEngine(query_engine=query_engine2, query_transform=step_decompose_transform)


async def vanilla_query_engine_tool(query: str):
    """This tool is useful for retrieving directly information from a vector database without any prior query transformation. It is mainly useful when the query is simple but specific"""
    response = await query_engine.aquery(query)
    return response.response

async def hyde_query_engine_tool(query: str):
    """This tool is useful for retrieving information from a vector database with the transformation of a query into an hypothetical document embedding, which will be used for retrieval. It is mainly useful when the query is general or vague."""
    response = await hyde_query_engine.aquery(query)
    return response.response

async def multi_step_query_engine_tool(query: str):
    """This tool is useful for retrieving information from a vector database with the decomposition of the query into a series of queries that will be iteratively executed against the vector database for retrieval. It is mainly useful when the query is complex and asks for nested and multi-faceted information."""
    response = await multistep_query_engine.aquery(query)
    return response.response


async def evaluate_context(original_prompt: str = Field(description="Original prompt provided by the user"), context: str = Field(description="Contextual information, either from retrieved documents")) -> str:
    """
    Useful for evaluating the coherence and relevance of retrieved contextual information in light of the user's prompt.

    This tool takes the original user prompt and contextual information as input, and evaluates the coherence of the response with the original prompt and the relevance of the contextual information. It returns a formatted string with the evaluation scores and reasons for the evaluations.

    Args:
        original_prompt (str): Original prompt provided by the user.
        context (str): Contextual information from retrieved documents.
    """
    messages = [ChatMessage.from_str(content=original_prompt, role="user"), ChatMessage.from_str(content=f"Here is some context that I found that might be useful for replying to the user:\n\n{context}", role="assistant"), ChatMessage.from_str(content="Can you please evaluate the relevance of the contextual information (giving it a score between 0 and 100) in light or my original prompt? You should also tell me the reasons for your evaluations.", role="user")]
    response = await llm_eval.achat(messages)
    json_response = json.loads(response.message.blocks[0].text)
    final_response = f"The context provided for the user's prompt is {json_response['context_is_ok']}% relevant.\nThese are the reasons why you are given these evaluations:\n{json_response['reasons']}"
    return final_response

async def evaluate_response(original_prompt: str = Field(description="Original prompt provided by the user"), context: str = Field(description="Contextual information, either from retrieved documents"), answer: str = Field(description="Final answer to the original prompt")) -> str:
    """
    Useful for evaluating the faithfulness and relevance of a response to a given prompt using contextual information.

    This tool takes an original prompt, contextual information, and a final answer, and evaluates the coherence of the response with the original prompt and the relevance of the contextual information. It returns a formatted string with the evaluation scores.

    Args:
        original_prompt (str): Original prompt provided by the user.
        context (str): Contextual information, either from retrieved documents or from the web, or both.
        answer (str): Final answer to the original prompt.
    """
    faithfulness = await faith_eval.aevaluate(query=original_prompt, response=answer, contexts=[context])
    relevancy = await rel_eval.aevaluate(query=original_prompt, response=answer, contexts=[context])
    rel_score = relevancy.score if relevancy.score is not None else 0
    fai_score = faithfulness.score if faithfulness.score is not None else 0
    return f"The relevancy of the produced answer is {rel_score*100}% and the faithfulness is {fai_score*100}%"


vanilla_rag_tool = FunctionTool.from_defaults(
    fn=vanilla_query_engine_tool,
    name="query_vanilla_rag",
    description="""This tool is useful for retrieving directly information from a vector database without any prior query transformation. It is mainly useful when the query is simple but specific
    
    Args:
        query (str): Query to search the vector database"""
)

hyde_rag_tool = FunctionTool.from_defaults(
    fn=hyde_query_engine_tool,
    name="query_hyde_rag",
    description="""This tool is useful for retrieving information from a vector database with the transformation of a query into an hypothetical document embedding, which will be used for retrieval. It is mainly useful when the query is general or vague.
    
    Args:
        query (str): Query to search the vector database"""
)


multistep_rag_tool = FunctionTool.from_defaults(
    fn=multi_step_query_engine_tool,
    name="query_multistep_rag",
    description="""This tool is useful for retrieving information from a vector database with the decomposition of the query into a series of queries that will be iteratively executed against the vector database for retrieval. It is mainly useful when the query is complex and asks for nested and multi-faceted information.
    
    Args:
        query (str): Query to search the vector database"""
)

evaluate_response_tool = FunctionTool.from_defaults(
    fn=evaluate_response,
    name="evaluate_response",
    description="""
Useful for evaluating the faithfulness and relevance of a response to a given prompt using contextual information.

This tool takes an original prompt, contextual information, and a final answer, and evaluates the coherence of the response with the original prompt and the relevance of the contextual information. It returns a formatted string with the evaluation scores.

Args:
    original_prompt (str): Original prompt provided by the user.
    context (str): Contextual information, either from retrieved documents or from the web, or both.
    answer (str): Final answer to the original prompt.
"""
)

evaluate_context_tool = FunctionTool.from_defaults(
    fn=evaluate_context,
    name="evaluate_context",
    description="""
Useful for evaluating the coherence and relevance of retrieved contextual information in light of the user's prompt.

This tool takes the original user prompt and contextual information as input, and evaluates the coherence of the response with the original prompt and the relevance of the contextual information. It returns a formatted string with the evaluation scores and reasons for the evaluations.

Args:
    original_prompt (str): Original prompt provided by the user.
    context (str): Contextual information from retrieved documents.
"""
)