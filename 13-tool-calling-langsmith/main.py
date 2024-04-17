
from typing import Optional
from pydantic import BaseModel
import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langsmith import traceable

from cerebrium import get_secret
import requests
from cal import find_available_slots

class Item(BaseModel):
    prompt: str
    session_id: str


##LangSmith ENV variables
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_API_KEY']=get_secret("LANGCHAIN_API_KEY")


@tool
def get_availability(fromDate: str, toDate: str) -> float:
    """Get my calendar availability using the 'fromDate' and 'toDate' variables in the date format '%Y-%m-%dT%H:%M:%S.%fZ'"""

    url = "https://api.cal.com/v1/availability"
    params = {
        "apiKey": get_secret("CAL_API_KEY"),
        "username": "michael-louis-ioje1y",
        "dateFrom": fromDate,
        "dateTo": toDate
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        availability_data = response.json()
        available_slots = find_available_slots(availability_data, fromDate, toDate)
        return available_slots
    else:
        return {}

@tool
def book_slot(datetime: str, name: str, email: str, title: str, description: str) -> float:
    """Book a meeting on my calendar at the requested date and time using the 'datetime' variable. The 'datetime' variable should be specified in EST. Get a description about what the meeting is about and make a title for it"""
    url = "https://api.cal.com/v1/bookings"
    params = {
        "apiKey": get_secret("CAL_API_KEY"),
        "username": "michael-louis-ioje1y",
        "eventTypeId": "737039",
        "start": datetime,
        "responses": {
            "name": name,
            "email": email,
            "title": title,
            
            "metadata": {},
            "location": "Cerebrium HQ"
        },
        "timeZone": "America/New York",
        "language": "en",
        "title": title,
        "description": description,
    }
    response = requests.post(url, params=params)
    if response.status_code == 200:
        booking_data = response.json()
        print(booking_data)
        return booking_data
    else:
        print('error')
        print(response)
        return {}

prompt = ChatPromptTemplate.from_messages([
    ("system", "you're a helpful assistant managing the calendar of Michael Louis. You need to book appointments for a user based on available capacity and their preference. You need to find out if the user is: From Michaels team, a customer of Cerebrium or a friend or entreperneur. If the person is from his team, book a morning slot. If its a potential customer for Cerebrium, book a afternoon slot. If its a friend or entreperneur needing help or advice, book a night time slot. If none of these are available, book the earliest slot. Do not book a slot without asking the user what their preferred time is. Find out from the user, their name and email address."), 
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"), 
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [get_availability, book_slot]


llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=get_secret("OPENAI_API_KEY") )
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
demo_ephemeral_chat_history_for_chain = ChatMessageHistory()
conversational_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="chat_history",
)

@traceable
def predict(item, run_id, logger):
    item = Item(**item)

    output = conversational_agent_executor.invoke(
        {
            "input": item.prompt,
        },
        {"configurable": {"session_id": item.session_id}},
    )

    return {"result": output} # return your results

# if __name__ == "__main__":
#     while True:
#         user_input = input("Enter the input (or type 'exit' to stop): ")
#         if user_input.lower() == 'exit':
#             break
#         result = predict({"prompt": user_input, "session_id": "12345"}, "test", logger=None)
#         print(result)
