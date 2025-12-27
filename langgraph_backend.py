from langgraph.graph import StateGraph,START,END 
from langchain_groq import ChatGroq
from typing import TypedDict,Annotated,Literal
from langchain_core.messages import SystemMessage, HumanMessage,BaseMessage
from langgraph.graph.message import  add_messages
from dotenv import load_dotenv
from pydantic import BaseModel, Field 

from langgraph.checkpoint.memory import InMemorySaver

import operator


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


load_dotenv()
llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0

)

def chat_node(state: ChatState) ->ChatState:
    # take user query
    messages= state['messages']

    # send to llm
    response= llm.invoke(messages)

    # response store
    return {'messages': [response]}


checkpointer=InMemorySaver()
# checkpointer divides graph execution in checkpoints, and saves the state  at checkpoints

graph= StateGraph(ChatState)

graph.add_node('chat_node',chat_node)


graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)
chatbot= graph.compile(checkpointer=checkpointer)

 
 
