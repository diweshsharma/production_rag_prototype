import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.reranking_services import rerank_documents

def retrieve_node(state : AgentState):
    """
    Performs vector Search and semantic reranking for technical queries.
    """
    query = state["current_query"]

    with logfire.span("Knowledge retrieval"):
        logfire.info(f"Querying Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} documents from Qdrant")

        doc_contents = [doc['content'] for doc in raw_results]

        with logfire.span("semantic reranking"):
            reranked_docs = rerank_documents(query,doc_contents,top_n=5)
            logfire.info(f"Reranking complete . kept top 5 most relevant documents")

        formatted_docs = [f"CONTENT :{doc}" for doc in reranked_docs]

    return {
        "documents" : formatted_docs,
        "status" : f"Found technical context",
        "plan" : state["plan"] + ["Context Retriever"]
    }

        
        