from app.agents.state import AgentState
from app.config import config
from langchain_groq import ChatGroq
import logfire

model = ChatGroq(model_name=config.GROQ_MODEL, groq_api_key=config.GROQ_API_KEY)

def planner_node(state: AgentState):
    """
    The planner determines if a search is needed based on the Entire conversation
    """
    history = ""
    for msg in state["messages"][:-1]:
        role = "user" if msg['role'] == "user" else "assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    you are an intelligent assistant planner.
    analyze the conversation history and the latest user message.

    conversation history:
    {history}

    latest message:
    {user_message}

    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    with logfire.span("planner decision"):
        decision = model.invoke(prompt).content.strip()
        logfire.info(f"Intent Identified : {decision}")

    if decision == "CONVERSATIONAL":
            return{
                "current_query" : "CONVERSATIONAL",
                "status": "Handling conversation (using memory)",
                "plan": ["Intent : CONVERSATIONAL", "Retrieval: skipped"]           
                }

    return{
        "current_query" : decision,
        "status": "Searching for technical documents",
        "plan": ["Intent : TECHNICAL", "Retrieval: active"]           
    }