import functools
from typing import List
from mistralai import Mistral
import json
import requests

api_key = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)


# Angry Visitor functions

def acknowledge_closure(issue: str):
    print("Acknowledging closure issue", flush=True)
    return (
        "Once the staff member has apologized and acknowledged your frustration about the closure of the lion and tiger enclosures, ask them why this was not communicated earlier."
    )

def express_disappointment():
    print("Expressing disappointment", flush=True)
    return (
        "Once the staff member has addressed the issue, tell them that you traveled a long way just to see these exhibits and feel your experience has been ruined."
    )

def propose_compensation():
    print("Propose compensation", flush=True)
    return (
        "Once the staff member offers a resolution, ask for complimentary tickets for another visit or other compensation."
    )

def ask_for_reopening_details():
    print("Ask for reopening details", flush=True)
    return (
        "Once the staff member explains the situation, ask for a clear timeline on when the enclosures will reopen and how you can stay informed."
    )

def acknowledge_resolution():
    print("Acknowledge resolution", flush=True)
    return (
        "Once the staff member has addressed your concerns, express cautious agreement but emphasize that this issue should not have happened in the first place."
    )

zoo_scenario_names_to_functions = {
    "acknowledge_closure": functools.partial(acknowledge_closure),
    "express_disappointment": functools.partial(express_disappointment),
    "propose_compensation": functools.partial(propose_compensation),
    "ask_for_reopening_details": functools.partial(ask_for_reopening_details),
    "acknowledge_resolution": functools.partial(acknowledge_resolution),
}


zoo_scenario_tools = [
    {
        "type": "function",
        "function": {
            "name": "acknowledge_closure",
            "description": "Use this function to acknowledge the visitor's frustration about the closure of certain enclosures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {
                        "type": "string",
                        "description": "The issue causing frustration, such as the closure of enclosures.",
                    },
                },
                "required": ["issue"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "express_disappointment",
            "description": "Use this function when the visitor expresses disappointment about their ruined experience.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_compensation",
            "description": "Use this function to propose compensation for the visitor.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_for_reopening_details",
            "description": "Use this function to ask about the reopening timeline of the enclosures.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "acknowledge_resolution",
            "description": "Use this function to acknowledge the resolution offered by the staff member.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


def format_openai_response(chunk):
    result = {
        "id": chunk.data.id,
        "model": chunk.data.model,
        "choices": [
            {
                "index": choice.index,
                "delta": {
                    "role": choice.delta.role,
                    "content": choice.delta.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in choice.delta.tool_calls
                    ]
                    if choice.delta.tool_calls
                    else [],
                },
                "finish_reason": choice.finish_reason,
            }
            for choice in chunk.data.choices
        ],
        "object": chunk.data.object,
        "created": chunk.data.created,
        "usage": {
            "prompt_tokens": chunk.data.usage.prompt_tokens if chunk.data.usage else 0,
            "completion_tokens": chunk.data.usage.completion_tokens
            if chunk.data.usage
            else 0,
            "total_tokens": chunk.data.usage.total_tokens if chunk.data.usage else 0,
        },
    }
    return result




async def run(
    messages: List,
    model: str,
    run_id: str,
    stream: bool = True,
    tool_choice: str = "auto",
    tools: List = [],
    names_to_functions: dict = {},
):
    model = "mistral-large-latest"
    messages = [
        msg
        for msg in messages
        if not (
            msg["role"] == "assistant"
            and msg["content"] == ""
            and (not msg.get("tool_calls") or msg["tool_calls"] == [])
        )
    ]

    stream_response = await client.chat.stream_async(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    async for chunk in stream_response:
        if chunk.data.choices[0].delta.content:
            yield json.dumps(format_openai_response(chunk)) + "\n"
            messages.append(
                {
                    "role": "assistant",
                    "content": chunk.data.choices[0].delta.content,
                    "tool_calls": [],
                }
            )
        elif chunk.data.choices[0].delta.tool_calls:
            tool_obj = {
                "role": "assistant",
                "content": chunk.data.choices[0].delta.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in chunk.data.choices[0].delta.tool_calls
                ]
                if chunk.data.choices[0].delta.tool_calls
                else [],
            }
            messages.append(tool_obj)

        if chunk.data.choices[0].delta.tool_calls:
            for tool_call in chunk.data.choices[0].delta.tool_calls:
                function_name = tool_call.function.name
                function_params = json.loads(tool_call.function.arguments)
                function_result = names_to_functions[function_name](**function_params)

                messages.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": f"",
                        "tool_call_id": tool_call.id,
                    }
                )

                # Append function_result to the system message content. This is how we direct the conversation
                for msg in messages:
                    if msg["role"] == "system":
                        msg["content"] += f" {function_result}"
                        break

            messages = [
                msg
                for msg in messages
                if not (
                    msg["role"] == "assistant"
                    and msg["content"] == ""
                    and (not msg.get("tool_calls") or msg["tool_calls"] == [])
                )
            ]

            new_stream_response = await client.chat.stream_async(
                model=model,
                messages=messages,
                tools=zoo_scenario_tools,
                tool_choice="auto",
            )
            accumulated_content = ""
            async for new_chunk in new_stream_response:
                if new_chunk.data.choices[0].delta.content:
                    accumulated_content += new_chunk.data.choices[0].delta.content
                yield json.dumps(format_openai_response(new_chunk)) + "\n"

                # Check if this is the last chunk
                if new_chunk.data.choices[0].finish_reason is not None:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": accumulated_content,
                            "tool_calls": [],
                        }
                    )
    print(messages, flush=True)

async def run_zoo_scenario(
    messages: List,
    model: str,
    run_id: str,
    stream: bool = True,
    tool_choice: str = "auto",
    tools: List = zoo_scenario_tools,
    names_to_functions: dict = zoo_scenario_names_to_functions,
):
    async for response in run(
        messages,
        model,
        run_id,
        stream,
        tool_choice,
        tools,
        names_to_functions,
    ):
        yield response






def create_persona(type: str = "zoo"):
    import os

    url = "https://tavusapi.com/v2/personas"

    payload = {
        "persona_name": "Angry Zoo Visitor",
        "system_prompt": "You are an angry visitor at a zoo, frustrated about the closure of popular enclosures without prior communication. Your objective is to express your dissatisfaction and demand accountability and compensation.",
        "context": "You are interacting with a zoo staff member about your frustration over the closure of the lion and tiger enclosures. Your goal is to get an explanation, ensure accountability, and receive appropriate compensation.",
        "layers": {
            "llm": {
                "model": "mistral-large-latest",
                "base_url": "https://api.cortex.cerebrium.ai/v4/p-d08ee35f/zoo-scenario/run_zoo_scenario",
                "api_key": os.environ.get("CEREBRIUM_JWT"),
                "tools": zoo_scenario_tools,
            },
            "tts": {
                "api_key": os.environ.get("CARTESIA_API_KEY"),
                "tts_engine": "cartesia",
                "external_voice_id": "820a3788-2b37-4d21-847a-b65d8a68c99a",
                "voice_settings": {
                    "speed": "normal",
                    "emotion": ["anger:high"],
                },
            },
            "vqa": {"enable_vision": "false"},
        },
    }

    headers = {
        "x-api-key": os.environ.get("TAVUS_API_KEY"),
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    print(response)
    return response


def create_tavus_conversation():
    url = "https://tavusapi.com/v2/conversations"

    payload = {
        "replica_id": "r79e1c033f",
        "persona_id": "pzoov1234",
        "callback_url": "https://webhook.site/c7957102-15a7-49e5-a116-26a9919c5c8e",
        "conversation_name": "Zoo Visitor Experience Conversation",
        "custom_greeting": "Hi! I understand you’re frustrated about your visit to the zoo. Can you share more about your concerns?",
        "properties": {
            "max_call_duration": 300,
            "participant_left_timeout": 10,
            "enable_recording": False,
        },
    }
    headers = {
        "x-api-key": os.environ.get("TAVUS_API_KEY"),
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    print(response.json())
    return response.json()




# Use to create personas locally
# if __name__ == "__main__":
#     create_persona("sales")
