import openai
from openai import OpenAI

def get_system_message():
    return "Please analyze the following text related to displacement and migration for factual accuracy and language appropriateness. Your task is to analyze the given comment_text and determine if it contains elements of toxicity, severe toxicity, obscenity, threat, insult, or identity hate. Provide the results in strict JSON format with each category labeled as 0 or 1."


client = OpenAI(
        api_key="")

system_message = get_system_message()
comment_text = "landing gear from one of the aircraft and bits and pieces of several people were found inside this building, which itself is less than 60 feet from another that was destroyed"
completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::AXY9Wf7H",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": comment_text}
        ]
    )
print(completion.choices[0].message.content)
