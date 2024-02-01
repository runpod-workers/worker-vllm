import os
from openai import OpenAI
client = OpenAI(
    api_key="XYZ",
    base_url="http://0.0.0.0:8000/v1"
)


stream = client.chat.completions.create(
  model="mistralai/Mistral-7B-Instruct-v0.1",
  messages=[
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played? Write a 1000 word essay about this"}
  ],
  max_tokens=300,
  stream=True,
)


for chunk in stream:
    print(chunk.choices[0].delta.content, end="")

# When you send a request to /openai/v1/chat/completions or to /openai/v1/completions 
# and you set the stream parameter to true, we need SSE streaming of the content