import json
from openai import OpenAI
import pandas as pd
import ast

client = OpenAI(api_key="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")

input_csv = "emojify_cleaned_new.csv"  # Replace with the path to your input CSV file
df = pd.read_csv(input_csv)

# client.api_key = "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"

sentiment_system_prompt = '''
Your goal is to extract the sentiment and emotional category of a sentence. You will be provided with a sentence, and you will output a json object containing the following information:

{
    sentiment: int // Sentiment of the sentence, where 1 is positive, -1 is negative, and 0 is neutral
    emotion: string[] // Array of categories of the sentence, must  be one of "joy", "sadness", "anger", "fear", "love", "surprise", "disgust"
}

Emotions refer to the emotional category of the sentence. Sentences can have multiple emotions, but try to keep it under 3-4 and at least 1. Only mention the emotions that are the most obvious based on the sentence.
'''

def get_labels(description):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.1,
    # This is to enable JSON mode, making sure responses are valid json objects
    response_format={ 
        "type": "json_object"
    },
    messages=[
        {
            "role": "system",
            "content": sentiment_system_prompt
        },
        {
            "role": "user",
            "content": description
        }
    ],
    )
    return response.choices[0].message.content

def convert_to_sentences(row):
    # Step 1: Convert the string representation of a list to an actual list
    words_list = ast.literal_eval(row)

    # Step 2: Join the words in the list into a sentence
    sentence = ' '.join(words_list)

    # breakpoint()
    return sentence

# Testing on a few examples
for _, row in df[:5].iterrows():
    sentence = convert_to_sentences(row['tokens_2'])
    result = get_labels(sentence)
    print(f"SENTENCE: {sentence}\nRESULT: {result}")
    print("\n\n----------------------------\n\n")


# Creating an array of json tasks

tasks = []

for index, row in df.iterrows():
    
    description = convert_to_sentences(row['tokens_2'])
    
    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": sentiment_system_prompt
                },
                {
                    "role": "user",
                    "content": description
                }
            ],
        }
    }
    
    tasks.append(task)

# Creating the file

file_name = "data/batch_tasks_movies.jsonl"

with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')

batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)

print(batch_file)


batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)
