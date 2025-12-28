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
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from datetime import datetime
from dateutil import parser as date_parser

import pytz
import sqlite3
import requests
import operator



load_dotenv()

# llm
llm=ChatGroq(
    model="llama-3.1-8b-instant",
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




class EventInfo(BaseModel):
    title: str = Field(description="Short event title")
    date: str = Field(description="Date like 'December 31 2025'")
    start_time: str = Field(description="Start time like '10 pm'")
    end_time: str = Field(description="End time like '10:30 pm'")


event_parser = PydanticOutputParser(pydantic_object=EventInfo)

event_prompt = PromptTemplate(
    template="""
Extract calendar event details from the sentence.

Sentence:
{text}

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": event_parser.get_format_instructions()
    }
)


IST = pytz.timezone("Asia/Kolkata")

def to_iso_datetime(date_text: str, time_text: str) -> str:
    """
    Converts natural language date & time into strict ISO datetime.
    Prevents minute/second leakage from default datetime.
    """

    now = datetime.now(IST)

    # Use a clean default (00:00) instead of "now"
    clean_default = now.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    dt = date_parser.parse(
        f"{date_text} {time_text}",
        default=clean_default
    )

    # Ensure future date
    if dt < now:
        dt = dt.replace(year=now.year + 1)

    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = IST.localize(dt)

    return dt.strftime("%Y-%m-%dT%H:%M:%S")



SCOPES = ["https://www.googleapis.com/auth/calendar"]
@tool
def create_calendar_event(text: str) -> dict:
    """
    Create a Google Calendar event from natural language text.
    Example:
    'Add event on December 31 at 10 pm till 10:30 pm called Year Ends'
    """

    # 1️⃣ LLM understands the text
    chain = event_prompt | llm | event_parser
    info: EventInfo = chain.invoke({"text": text})

    # 2️⃣ Python converts to ISO
    start_iso = to_iso_datetime(info.date, info.start_time)
    end_iso = to_iso_datetime(info.date, info.end_time)

    # 3️⃣ Google Calendar API
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    service = build("calendar", "v3", credentials=creds)

    event = {
        "summary": info.title,
        "start": {
            "dateTime": start_iso,
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": end_iso,
            "timeZone": "Asia/Kolkata",
        },
    }

    created = service.events().insert(
        calendarId="primary",
        body=event
    ).execute()

    return {
        "status": "created",
        "title": info.title,
        "start": start_iso,
        "end": end_iso,
        "link": created.get("htmlLink")
    }
# tool list
tools= [search_tool, calculator, get_stock_price,create_calendar_event]
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