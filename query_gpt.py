from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful and creative assistant who can understand analogies, relate concepts to each other, and outputs a JSON string."},
    {"role": "user", "content": "If atom is like solar system, then nucleus is like"}
  ]
)

print(response)

print(f'{response["usage"]["prompt_tokens"]} prompt tokens counted by the OpenAI API.')