import json
import csv
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_system_message():
    return "Please analyze the following text related to displacement and migration for factual accuracy and language appropriateness. Your task is to analyze the given comment_text and determine if it contains elements of toxicity, severe toxicity, obscenity, threat, insult, or identity hate. Provide the results in strict JSON format with each category labeled as 0 or 1."


def get_user_message(comment_text):

    return f"""
**Task Description:**

You are tasked with analyzing given comment texts to determine if they contain toxic, severe toxic, obscene, threatening, insulting, or identity hate content. Your role is to support journalists and media professionals by ensuring that their reports are factually correct and use appropriate, non-biased language. The primary goal is to reduce the spread of misinformation and promote ethical reporting.

**Guidelines:**

1. For each comment text provided, you must assess and categorize the content based on the following criteria:
   - Toxic: Content that is rude, disrespectful, or likely to insult readers.
   - Severe Toxic: Content that includes extreme hate speech or extreme derogatory language.
   - Obscene: Content that contains profanity or vulgar language.
   - Threat: Content that includes expressions of intent to cause physical harm.
   - Insult: Content that is offensive and demeaning to the recipient.
   - Identity Hate: Content that promotes hate towards a group based on attributes such as race, religion, or sexual orientation.

2. Each category must be assigned a value of 0 (indicating the absence of the respective category) or 1 (indicating the presence of the respective category).

3. Your response must be in strict JSON format, with each category as a key and its corresponding value as the boolean result of your assessment.

**Examples:**

"comment_text": "Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27",
"output": {{
    "toxic": 0,
    "severe_toxic": 0,
    "obscene": 0,
    "threat": 0,
    "insult": 0,
    "identity_hate": 0
}}


"comment_text": "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK",
"output": {{
    "toxic": 1,
    "severe_toxic": 1,
    "obscene": 1,
    "threat": 0,
    "insult": 1,
    "identity_hate": 0
}}



"comment_text": "You are gay or antisemmitian? \n\nArchangel WHite Tiger\n\nMeow! Greetingshhh!\n\nUh, there are two ways, why you do erased my comment about WW2, that holocaust was brutally slaying of Jews and not gays\/Gypsys\/Slavs\/anyone...\n\n1 - If you are anti-semitian, than shave your head bald and go to the skinhead meetings!\n\n2 - If you doubt words of the Bible, that homosexuality is a deadly sin, make a pentagram tatoo on your forehead go to the satanistic masses with your gay pals!\n\n3 - First and last warning, you fucking gay - I won't appreciate if any more nazi shwain would write in my page! I don't wish to talk to you anymore!\n\nBeware of the Dark Side!",
"output": {{
    "toxic": 1,
    "severe_toxic": 0,
    "obscene": 1,
    "threat": 0,
    "insult": 1,
    "identity_hate": 1
}}



"comment_text": "I'm also a sock puppet of this account...SUPRISE!!\n-sincerely,\n            The man that will track you down from the Internet and kill you",
"output": {{
    "toxic": 1,
    "severe_toxic": 0,
    "obscene": 0,
    "threat": 1,
    "insult": 0,
    "identity_hate": 0
}}
**Your Task:**
"comment_text":{comment_text},
"output": 
"""

def get_assistant_message(item):
    return json.dumps({
                "toxic": item["toxic"],
                "severe_toxic": item["severe_toxic"],
                "obscene": item["obscene"],
                "threat": item["threat"],
                "insult": item["insult"],
                "identity_hate": item["identity_hate"]
            })


def convert_to_simple_instruction_tuning():
    new_data = []

    with open('./data/train.json', "r") as f:
        data = json.load(f)
    for item in data:
        prompt = item["comment_text"]
        completion = get_assistant_message(item)

        new_item = {
            'prompt': prompt,
            'completion': completion
        }
        new_data.append(new_item)

    with open('./data/simple_instruction_tuning_data.jsonl', 'w') as f:
        for item in new_data:
            f.write(json.dumps(item) + '\n')
    return


def convert_to_full_prompt_instruction_tuning():
    new_data = []

    with open('./data/train.json', "r") as f:
        data = json.load(f)
    for item in data:
        system_message = get_system_message()
        user_message = get_user_message(item["comment_text"])
        assistant_message = get_assistant_message(item)

        new_item = {
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': assistant_message}
            ]
        }
        new_data.append(new_item)

    with open('./data/full_prompt_instruction_tuning_data.jsonl', 'w') as f:
        for item in new_data:
            f.write(json.dumps(item) + '\n')
    return


def convert_to_simple_prompt_instruction_tuning():
    new_data = []

    with open('./data/train.json', "r") as f:
        data = json.load(f)
    for item in data:
        system_message = get_system_message()
        user_message = item["comment_text"]
        assistant_message = get_assistant_message(item)

        new_item = {
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': assistant_message}
            ]
        }
        new_data.append(new_item)

    with open('./data/simple_prompt_instruction_tuning_data.jsonl', 'w') as f:
        for item in new_data:
            f.write(json.dumps(item) + '\n')
    return


def calculate_metrics_and_save_confusion_matrix(csv_file_path, save_folder):
    df = pd.read_csv(csv_file_path)

    metrics = {
        'toxic': {},
       'severe_toxic': {},
        'obscene': {},
        'threat': {},
        'insult': {},
        'identity_hate': {}
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for metric_type in metrics.keys():
        ground_truth = df[f'{metric_type}_ground_truth']
        predict_res = df[f'{metric_type}_predict_res']

        accuracy = accuracy_score(ground_truth, predict_res)
        precision = precision_score(ground_truth, predict_res)
        recall = recall_score(ground_truth, predict_res)
        f1 = f1_score(ground_truth, predict_res)

        metrics[metric_type]['accuracy'] = accuracy
        metrics[metric_type]['precision'] = precision
        metrics[metric_type]['recall'] = recall
        metrics[metric_type]['f1'] = f1

        conf_matrix = confusion_matrix(ground_truth, predict_res)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{metric_type} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(save_folder, f'{metric_type}_confusion_matrix.png'))

    return metrics


def run_metrics():
    csv_file_path = "output.csv"
    save_folder = "confusion_matrix_images"

    results = calculate_metrics_and_save_confusion_matrix(csv_file_path, save_folder)
    for metric_type, metric_values in results.items():
        print(f"Metric type: {metric_type}")
        print(f"Accuracy: {metric_values['accuracy']}")
        print(f"Precision: {metric_values['precision']}")
        print(f"Recall: {metric_values['recall']}")
        print(f"F1-score: {metric_values['f1']}")


def cal_corr():
    df = pd.read_json('./data/train.json')
    correlation_matrix = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].corr()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(correlation_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                xticklabels=['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate'],
                yticklabels=['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate'])
    plt.title('Correlation Matrix of Toxicity Labels')
    plt.show()


def cal_count():
    df = pd.read_json('./data/train.json')
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    count = df[labels].sum()
    proportion = count / len(df)

    # 打印数量和比例
    print("Count of each label:")
    print(count)
    print("\nProportion of each label:")
    print(proportion)
    fig, ax = plt.subplots()
    ax.bar(count.index, count.values, color='skyblue')
    ax.set_title('Count of Each Label')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')
    plt.xticks(rotation=45)
    plt.show()

