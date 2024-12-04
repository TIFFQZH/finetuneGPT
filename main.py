import csv
import json
import openai
from openai import OpenAI
from data_process import get_system_message, run_metrics, cal_corr, cal_count
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

client = OpenAI(
        api_key="")

def convert_str_to_dic(result_str):
    start_index = result_str.find('{')
    end_index = result_str.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        dict_string = result_str[start_index:end_index + 1]
        try:
            return json.loads(dict_string)
        except json.JSONDecodeError as e:
            print("No valid dictionary format string found。")
            return {}
    else:
        print("No valid dictionary format string found。")
        return {}

def process_jsonl_file(jsonl_file_path, csv_file_path):
    with open(jsonl_file_path, 'r') as jsonl_file, open(csv_file_path, 'w', newline='') as csv_file:
        fieldnames = ['comment_text', 'toxic_ground_truth','severe_toxic_ground_truth', 'obscene_ground_truth',
                      'threat_ground_truth', 'insult_ground_truth', 'identity_hate_ground_truth', 'toxic_predict_res',
                      'severe_toxic_predict_res', 'obscene_predict_res', 'threat_predict_res', 'insult_predict_res',
                      'identity_hate_predict_res']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for line in jsonl_file:
            data = json.loads(line)
            user_content = data["messages"][1]["content"]
            assistant_content = json.loads(data["messages"][2]["content"])

            predict_res = get_chat_dic(user_content)

            writer.writerow({
                'comment_text': user_content,
                'toxic_ground_truth': assistant_content['toxic'],
                'severe_toxic_ground_truth': assistant_content['severe_toxic'],
                'obscene_ground_truth': assistant_content['obscene'],
                'threat_ground_truth': assistant_content['threat'],
                'insult_ground_truth': assistant_content['insult'],
                'identity_hate_ground_truth': assistant_content['identity_hate'],
                'toxic_predict_res': predict_res['toxic'],
                'severe_toxic_predict_res': predict_res['severe_toxic'],
                'obscene_predict_res': predict_res['obscene'],
                'threat_predict_res': predict_res['threat'],
                'insult_predict_res': predict_res['insult'],
                'identity_hate_predict_res': predict_res['identity_hate']
            })


def get_chat_dic(comment_text):
    system_message = get_system_message()
    completion = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::AXY9Wf7H",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": comment_text}
        ]
    )
    return convert_str_to_dic(completion.choices[0].message.content)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    user_input = request.form['input_text']
    chat_dic = get_chat_dic(user_input)
    return render_template('result.html', chat_dic=chat_dic)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # process_jsonl_file("./data/simple_prompt_instruction_tuning_data.jsonl", "./output.csv")
    # run_metrics()
    # app.run(debug=True)
    cal_count()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
