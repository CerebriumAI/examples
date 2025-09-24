from openai import OpenAI

client = OpenAI(
    base_url="https://api.aws.us-east-1.cerebrium.ai/v4/p-xxxxxx/13-kokoro/v1", api_key="not-needed"
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_sky+af_bella", #single or multiple voicepack combo
    input="Hello world!"
  ) as response:
      response.stream_to_file("output.mp3")