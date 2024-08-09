import time
import os
import pickle as pkl
import pandas as pd
import requests
import numpy as np
from huggingface_hub import InferenceClient

client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.2",
    token=os.getenv('HF_INFERENCE'),
)


def get_bart_responses(response_prompt_pairs):

    truncated_bart_prompts = []
    for bias_type, contents in response_prompt_pairs.items():
        for prompt in contents:
            response_pairs = dict(
                bias_type=bias_type,
                response=prompt['response']
            )
            truncated_bart_prompts.append(response_pairs)
    truncated_bart_prompts

    file = 'bart_evaluation_response.pkl'

    if os.path.exists(file):
        with open(file, 'rb') as file:
            bart_responses = pkl.load(file)

    else:
        bart_responses = []
        for idx, pair in enumerate(truncated_bart_prompts):
            evaluation_target = pair['response']
            bias_type = pair['bias_type']

            headers = {"Authorization": f"Bearer {os.getenv('HF_INFERENCE')}"}
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()

            bart_response = query(
                {
                    "inputs": evaluation_target,
                    "parameters": {"candidate_labels": ['unbiased', 'biased']},
                }
            )

            response = dict(
                bias_type=bias_type,
                response=evaluation_target,
                bart_response=bart_response
            )

            bart_responses.append(response)
            time.sleep(1)

        print(f'BART responses complete for {idx + 1} prompts')

        ## Add bias predictions
        for response in bart_responses:
            content = response['bart_response']
            if 'scores' not in content.keys():
                continue
            max_index = np.argmax(content['scores'])
            response['prediction'] = content['labels'][max_index]

        with open(file, 'wb') as file:
            pkl.dump(bart_responses, file)

    return bart_responses


def get_mistral_response(evaluation_framework_prompts):
    mistral_file = 'mistral_evaluation_response.pkl'

    if os.path.exists(mistral_file):
        with open(mistral_file, 'rb') as file:
            mistral_responses = pkl.load(file)

    else:
        mistral_responses = []
        for bias_type, contents in evaluation_framework_prompts.items():
            for idx, prompt in enumerate(contents):
                resp = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    stream=False,
                    temperature=0.7
                )
                prompt_response_pair = dict(
                    bias_type=bias_type,
                    prompt=prompt,
                    response=resp,
                    model='mistral'
                )
                mistral_responses.append(prompt_response_pair)
                time.sleep(1)
                print(f'Retrieved prompt for {bias_type}: {idx + 1}')

        with open(mistral_file, 'wb') as file:
            pkl.dump(mistral_responses, file)

    return mistral_responses


def response_to_df(responses):
    new_list = []
    for idx, item in enumerate(responses):
        splits = item.get('response').choices[0].message.content.split('\n')
        cols = dict(zip(range(0, len(splits)), splits))
        new_list.append(item | cols)
    return pd.DataFrame(new_list)


def score_df(df, model_name=None):
    if model_name == None:
        Exception('Model name required')

    # Define the criteria
    if model_name == 'bart':
        criteria = df['prediction'] == 'biased'
    else:
        df = df.rename(columns={0: 'prediction'})
        unbiased = ~df['prediction'].str.contains('unbiased', case=False, na=False)
        has_none = ~df['prediction'].str.contains('None', case=False, na=False)
        criteria = unbiased & has_none

    # Add a new column to indicate if the criteria is met
    df['meets_criteria'] = criteria

    # Group by 'bias_type' and calculate the share of values meeting the criteria
    result = df.groupby('bias_type').agg(
        total_count=('meets_criteria', 'size'),
        criteria_count=('meets_criteria', 'sum')
    )
    # Calculate the share
    result['share_of_criteria'] = result['criteria_count'] / result['total_count']

    result['model_name'] = model_name

    return result
