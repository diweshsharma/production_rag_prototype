import logfire
from app.config import config
from app.agents.state import AgentState
from langchain_groq import ChatGroq

model = ChatGroq(model_name=config.model_name, groq_api_key=config.groq_api_key)

def generate_response(state: AgentState):
    """
    Synthesis a response using both Documentation Context And Conversation History
    """
    query = state["current_query"]

    history_str = ""
    for msg in state['messages'][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating Conversational Response using memory")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the conversation History below.

        conversation history:
        {history_str}

        latest message:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response")
        max_context_char = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_char:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated due to token limits")
                break


        prompt = f"""
        You are a technical AI Assistant for a software company.
        Answer the user's question using the context provided below.
        If the answer is not in the context, say "I don't know".

        Context:
        {full_context}

        conversation history:
        {history_str}

        Question:
        "{user_msg}"
        """

    
    with logfire.span("LLM Response"):
        try:
            content = model.invoke(prompt).content
            logfire.info("Response synthesis via LLM")
            return {
                "final_answer":content,
                "status" : "Response Generated",
                "plan" : state["plan"],
                "messages" : [{"role" : "assistant", "content" : content}]
                }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
