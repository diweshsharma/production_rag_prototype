from typing import Annotated , TypedDict , List 
import operator

class AgentState(TypedDict):
    #using annotated with operator .add ensures that messages
    #are appended to the history rather than replaced
    messages: Annotated[List[dict], operator.add]
    current_query: str
    documents: List[str]
    plan : List[dict]
    status: str
    final_answer: str

    

