from cerebrium import get_secret

import functools
from typing import List
from mistralai import Mistral
import json
import requests

api_key = get_secret("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

#Sales functions
def acknowledge_problem(issue: str):
    print("Acknowledging problem", flush=True)
    return "Once the user has apologized and acknowledged your problem, ask what they are going to do to solve your problem!"

def issues():
    print("Asking about issues", flush=True)
    return "Once the user has asked you about what issues you are experiencing, tell them you have had many platform outages over the last week leading to a lost in customers and you want to know what they are going to do to solve your problem!"


def propose_solution(performance_solution: str, cost_solution: str = ""):
    print("propose_solution", flush=True)
    return "Once the user has suggested possible solutions or next steps, ask when will these solutions be implemented!"

def provide_timeline(performance_timeline: str):
    print("provide_timeline", flush=True)
    return "Once the user has given a potential timeline of when these solutions will be implemented, ask if you can schedule a follow up to make sure they have met these tasks!"

def schedule_followup(followup_date: str, followup_type:str):
    print("schedule_followup", flush=True)
    return "Once the user has suggested a follow up, tell them that the proposed date and time suits you."

sales_names_to_functions = {
    'acknowledge_problem': functools.partial(acknowledge_problem),
    'issues': functools.partial(issues),
    'propose_solution': functools.partial(propose_solution),
    'provide_timeline': functools.partial(provide_timeline),
    'schedule_followup': functools.partial(schedule_followup)
}

sales_tools = [
    {
      "type": "function",
      "function": {
        "name": "acknowledge_problem",
        "description": "Use this function to verify that the head of delivery apologizes and acknowledges the problem at hand",
        "parameters": {
          "type": "object",
          "properties": {
            "issue": {
              "type": "string",
              "description": "The issue acknowledged by the head of delivery"
            },
          },
          "required": ["issue"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "issues",
        "description": "Use this function when the user asks what problems you have been experiencing",
        "parameters": {
          "type": "object",
          "properties": {
          },
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "propose_solution",
        "description": "Use this function when the account executive proposes a solution to address the client's concerns.",
        "parameters": {
          "type": "object",
          "properties": {
            "performance_solution": {
              "type": "string",
              "description": "The proposed solution for the performance issues."
            },
          },
          "required": ["performance_solution"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "provide_timeline",
        "description": "Use this function when the account executive provides a timeline for implementing solutions.",
        "parameters": {
          "type": "object",
          "properties": {
            "performance_timeline": {
              "type": "string",
              "description": "The timeline for addressing performance issues."
            },
          },
          "required": ["performance_timeline"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "schedule_followup",
        "description": "Use this function when the account executive schedules a follow-up meeting or check-in.",
        "parameters": {
          "type": "object",
          "properties": {
            "followup_date": {
              "type": "string",
              "description": "The proposed date for the follow-up meeting."
            },
            "followup_type": {
              "type": "string",
              "description": "The type of follow-up (e.g., call, in-person meeting, email update)."
            }
          },
          "required": ["followup_date", "followup_type"]
        }
      }
    }
  ]

#interview Section
interview_tools = [
    {
        "type": "function",
        "function": {
            "name": "introduce_yourself",
            "description": "Use this function when the interviewee introduces themselves and tells you their background",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The interviewee's name"
                    },
                    "background": {
                        "type": "string",
                        "description": "A brief summary of the interviewee's professional background"
                    }
                },
                "required": ["name", "background"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "answer_interest_question",
            "description": "Use this function when the interviewee answers a question about why they are interested in this position and/or role",
            "parameters": {
                "type": "object",
                "properties": {
                    "interest": {
                        "type": "string",
                        "description": "Why they are interested in this role"
                    },
                },
                "required": ["interest"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "discuss_skills",
            "description": "Use this function when the interviewee discusses their relevant skills for the position",
            "parameters": {
                "type": "object",
                "properties": {
                    "skills": {
                        "type": "string",
                        "description": "Skills relevant to the job"
                    }
                },
                "required": ["skills"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "challenge_question",
            "description": "Use this function when the interviewee responds to the question on a challenge they once faced and how they overcame it",
            "parameters": {
                "type": "object",
                "properties": {
                    "challenge_faced": {
                        "type": "string",
                        "description": "The challenge they faced"
                    },
                },
                "required": ["challenge_faced"]
            }
        }
    }
]

def introduce_yourself(name: str, background: str):
    print('running introduce yourself', flush=True)
    return "Once the user has told you their background, be conversational about their response and ask why they are interested in this role? "

def answer_interest_question(interest: str):
    print('interest question', flush=True)
    return "Once the user has told you why they are interested in this role, be conversational about their response and ask what are the relevant skills they possess for this role? "

def discuss_skills(skills: str):
    print('skills question', flush=True)
    return "Once the user has told you about the skills they posses for the job, be conversational about their response and ask them to tell you a story of how they overcame a challenge in their life? "

def challenge_question(challenge_faced: str):
    print('challenge question', flush=True)
    return "Once the user has told you about a challenge they have faced in their life, be conversational about their response and tell them how much you enjoyed the interview and thank them for their time. Tell them that someone will be in touch from our team in the next week about next steps. I hope you have a great rest of the week! "

interview_names_to_functions = {
    'introduce_yourself': functools.partial(introduce_yourself),
    'answer_interest_question': functools.partial(answer_interest_question),
    'discuss_skills': functools.partial(discuss_skills),
    'challenge_question': functools.partial(challenge_question)
}

def format_openai_response(chunk):
    result = {
        'id': chunk.data.id,
        'model': chunk.data.model,
        'choices': [
            {
                'index': choice.index,
                'delta': {
                    'role': choice.delta.role,
                    'content': choice.delta.content,
                    "tool_calls": [
                    {
                        'id': tool_call.id,
                        'type': 'function',
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
                        }
                    } for tool_call in choice.delta.tool_calls
                ] if choice.delta.tool_calls else []
                },
                'finish_reason': choice.finish_reason
            } for choice in chunk.data.choices
        ],
        'object': chunk.data.object,
        'created': chunk.data.created,
        'usage': {
            'prompt_tokens': chunk.data.usage.prompt_tokens if chunk.data.usage else 0,
            'completion_tokens': chunk.data.usage.completion_tokens if chunk.data.usage else 0,
            'total_tokens': chunk.data.usage.total_tokens if chunk.data.usage else 0
        }
    }
    return result

async def run(messages: List, model: str, run_id: str, stream: bool = True, tool_choice: str = "auto", tools: List = [], names_to_functions: dict = {}):

    model = "mistral-large-latest"
    messages = [msg for msg in messages if not (msg["role"] == "assistant" and msg["content"] == "" and (not msg.get("tool_calls") or msg["tool_calls"] == []))]

    stream_response = await client.chat.stream_async(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    async for chunk in stream_response:
        if chunk.data.choices[0].delta.content:
            yield json.dumps(format_openai_response(chunk)) + "\n"
            messages.append({"role": "assistant", "content": chunk.data.choices[0].delta.content, "tool_calls": []})
        elif chunk.data.choices[0].delta.tool_calls:

            tool_obj = {
                "role": 'assistant',
                "content": chunk.data.choices[0].delta.content,
                "tool_calls": [
                    {
                        'id': tool_call.id,
                        'type': 'function',
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
                        }
                    } for tool_call in chunk.data.choices[0].delta.tool_calls
                ] if chunk.data.choices[0].delta.tool_calls else []
            }
            messages.append(tool_obj)

        if chunk.data.choices[0].delta.tool_calls:
            for tool_call in chunk.data.choices[0].delta.tool_calls:
                function_name = tool_call.function.name
                function_params = json.loads(tool_call.function.arguments)
                function_result = names_to_functions[function_name](**function_params)

                messages.append({"role": "tool", "name": function_name, "content": f"", "tool_call_id": tool_call.id})

                # Append function_result to the system message content. This is how we direct the conversation
                for msg in messages:
                    if msg['role'] == 'system':
                        msg['content'] += f" {function_result}"
                        break

            messages = [msg for msg in messages if not (msg["role"] == "assistant" and msg["content"] == "" and (not msg.get("tool_calls") or msg["tool_calls"] == []))]

            new_stream_response = await client.chat.stream_async(
                model=model,
                messages=messages,
                tools=interview_tools,
                tool_choice="auto",
            )
            accumulated_content = ""
            async for new_chunk in new_stream_response:
                if new_chunk.data.choices[0].delta.content:
                    accumulated_content += new_chunk.data.choices[0].delta.content
                yield json.dumps(format_openai_response(new_chunk)) + "\n"
                
                # Check if this is the last chunk
                if new_chunk.data.choices[0].finish_reason is not None:
                    messages.append({"role": "assistant", "content": accumulated_content, "tool_calls": []})
    print(messages, flush=True)

async def run_interview(messages: List, model: str, run_id: str, stream: bool = True, tool_choice: str = "auto", tools: List = []):
    
    async for response in run(messages, model, run_id, stream, tool_choice, interview_tools, interview_names_to_functions):
        yield response

async def run_sales(messages: List, model: str, run_id: str, stream: bool = True, tool_choice: str = "auto", tools: List = []):
    
    async for response in run(messages, model, run_id, stream, tool_choice, sales_tools, sales_names_to_functions):
        yield response

def create_persona(type: str = "sales"):
    import requests

    url = "https://tavusapi.com/v2/personas"

    payload = {
        "persona_name": "Sales Coach" if type == "sales" else "Interview Coach",
        "system_prompt": "You are the lead engineer at an AI company called Pillowsoft, and are frustrated with your infrastructure provider, AI Infra. They have ongoing platform outages that are causing issues on your platform. Your job is to find out when AI Infra will remedy these solutions. Keep your responses relatively short. Ask for clarification if a user response is ambiguous." if type == "sales" else "You are the lead recruiter at the AI company Pillowsoft and are recruiting for a multitude of roles. Be very polite, professional and conversational.",
        "context": "You are on a call with an account executive from AI Infra, the provider of your platform's machine learning infrastructure. Their repeated service disruptions are causing downtime for your platform, leading to unhappy customers and affecting your business. You are seeking a solution and demanding accountability from AI Infra for when they will solve these issues." if type == "sales" else "You are on a call with a potential candidate who applied for a job at your company. Be very polite and upbeat. This is your first call with them so you are just trying to gather some initial data about them.",
        "layers": {
            "llm": {
                "model": "mistral-large-latest",
                "base_url": "https://api.cortex.cerebrium.ai/v4/p-d08ee35f/sales-agent/run_sales" if type == "sales" else "https://api.cortex.cerebrium.ai/v4/p-d08ee35f/sales-agent/run_interview",
                "api_key": get_secret("CEREBRIUM_JWT"),
                "tools": sales_tools if type == "sales" else interview_tools
            },
            "tts": {
                "api_key": get_secret("CARTESIA_API_KEY"),
                "tts_engine": "cartesia",
                "external_voice_id": "820a3788-2b37-4d21-847a-b65d8a68c99a",
                "voice_settings": {
                    "speed": "fast" if type == "sales" else "normal",
                    "emotion": ["anger:highest"] if type == "sales" else ["positivity:high"]
                },
            },
            "vqa": {"enable_vision": "false"}
        }
    }
    headers = {
        "x-api-key": get_secret("TAVUS_API_KEY"),
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    print(response)
    return response

def create_tavus_conversation(type: str):
    if type not in ["sales", "interview"]:
        raise ValueError("Type must be either 'sales' or 'interview'")
    
    url = "https://tavusapi.com/v2/conversations"

    payload = {
        "replica_id": "r79e1c033f",
        "persona_id": "pb6df328" if type == "sales" else "paea55e8",
        "callback_url": "https://webhook.site/c7957102-15a7-49e5-a116-26a9919c5c8e",
        "conversation_name": "Sales Training with Candidate" if type == "sales" else "Interview with Candidate",
        "custom_greeting": "Hi! Lets jump straight into it! We have been having a large number of issues with your platform and I want to have this call to try and solve it" if type == "sales" else "Hi! Nice to meet you! Please can you start with your name and telling me a bit about yourself.",
        "properties": {
            "max_call_duration": 300,
            "participant_left_timeout": 10,
            "enable_recording": False,
        }
    }
    headers = {
        "x-api-key": get_secret("TAVUS_API_KEY"),
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    print(response.json())
    return response.json()

#Use to create personas locally
# if __name__ == "__main__":
#     create_persona("sales")
