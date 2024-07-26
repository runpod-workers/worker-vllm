from openai import OpenAI
import os

# Initialize the OpenAI Client with your RunPod API Key and Endpoint URL
client = OpenAI(
    api_key="E43IAWSFPQPHJ8WIVLQ7EMBJ2TV9T3HF5YBLJJ7K",
    base_url="https://api.runpod.ai/v2/6ruq7l9hptgccv/openai/v1/",
)


# Create a chat completion
response = client.chat.completions.create(
    model="openchat/openchat-3.5-1210",
    messages=[{"role": "user", "content": "Why is RunPod the best platform?"}],
    temperature=0,
    max_tokens=100,
)
# Print the response
print(response.choices[0].message.content)


# Create a completion
# response = client.completions.create(
#     model="openchat/openchat-3.5-1210",
#     prompt="Runpod is the best platform because",
#     temperature=0,
#     max_tokens=100,
# )
# # Print the response
# print(response.choices[0].text)