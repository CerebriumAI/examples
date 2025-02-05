from openai import OpenAI
# Initialize client pointing to local endpoint
client = OpenAI(
    base_url="https://api.cortex.cerebrium.ai/v4/p-c6754f15/llama/run",
    api_key="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWM2NzU0ZjE1IiwiaWF0IjoxNzE4MzcyMDI1LCJleHAiOjIwMzM5NDgwMjV9.FkqWD4wrXWxWBzUolH5APtv5bMgClYt0ZbYnRV_WUTvZXOsynHvA3FhdSPiLiadfai9HwtWq8pGXazNCZSUs1xovxiW08oFwu1yiFIHpA_j64tMoqAmH0_kd4-PpGWTeyYognlQfr63eFvsv_Gpab2_kdt6JuZel-zOJWsaVwCPKAT02JWX7xhfzSoqyT_WdE0J2pJH8mFWcVJjL9f5YpQfgg8oJlPS1fm47sHfEoSVm2Usd3pQuouVAmEkPjttpwBep5uTNIEDqyCrCNSWCHJwvpSqlw7oOJoOuryuCQu1klJd3Fbw95FUg1Qg_NZLMb3pAZkR2eXThHz554PMLVw" # API key isn't actually used but required by SDK
)

# Example messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short poem about coding."}
]

# Stream the completion
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=messages,
    stream=True,
    max_tokens=128,
    temperature=0.8
)

# Print streamed responses
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print() # New line at the end
