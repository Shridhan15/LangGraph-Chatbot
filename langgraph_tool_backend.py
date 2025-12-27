from langgraph.graph import StateGraph,START,END 
from langchain_groq import ChatGroq
from typing import TypedDict,Annotated,Literal
from langchain_core.messages import SystemMessage, HumanMessage,BaseMessage
from langgraph.graph.message import  add_messages
from dotenv import load_dotenv
from pydantic import BaseModel, Field 

from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.runnables.graph import MermaidDrawMethod

import sqlite3
import requests
import operator



load_dotenv()

# llm
llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0

)

# tools
# tools
search_tool= DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers.
    Supported operations add, sub, mul, div"""

    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {'error':"Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {'error':f"unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {'error': str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """ Fetch latest stock price for a given symbol (e.g 'APPL','TSLA')
    using Finnhub with API key in URL"""
    url= f"https://finnhub.io/api/v1/quote?symbol={symbol}&token=d56m1g1r01qkgd82gq4gd56m1g1r01qkgd82gq50"
    # d56m1g1r01qkgd82gq4gd56m1g1r01qkgd82gq50
    r= requests.get(url)
    return r.json()


# tool list
tools= [search_tool, calculator, get_stock_price]
# make llm with tools
llm_with_tools= llm.bind_tools(tools)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



def chat_node(state: ChatState) ->ChatState:
    # take user query
    messages= state['messages']

    # send to llm
    response= llm_with_tools.invoke(messages)

    # response store
    return {'messages': [response]}

tool_node= ToolNode(tools)


# we use sqlite saver as DB to store messages
conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)
# checkpointer divides graph execution in checkpoints, and saves the state  at checkpoints

graph= StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)


graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools',"chat_node")
chatbot= graph.compile(checkpointer=checkpointer)


# get no of threads
def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)