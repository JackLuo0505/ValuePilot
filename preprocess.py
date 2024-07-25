"""
Preprocess data for action-value modeling.
Author: Anonymous Author
Date: ---

This script preprocesses the data for action-value modeling. It includes functions to generate input texts, preprocess individual samples, and preprocess entire datasets.

In this script:
- get_input_text_list: Generates a list of input texts based on the configuration.
- random_select_sample: Randomly selects a sample from the dataset.
- preprocess_sample: Preprocesses a single sample, including tokenization and encoding.
- preprocess: Preprocesses the entire dataset, including tokenization and encoding of all samples.
"""

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import itertools
import os
import random
import argparse
from sklearn.metrics import f1_score

def get_input_text_list(num_value, dataset='gpt4', mode='test', value_dimensions=['curiosity', 'energy', 'safety', 'happiness', 'intimacy', 'fairness']):
    """
    Generates a list of input texts based on the configuration.

    Args:
    - num_value (int): Number of value dimensions to consider.
    - dataset (str): Dataset identifier.
    - mode (str): Mode of the dataset (train, test, validation).
    - value_dimensions (list): List of value dimensions.

    Returns:
    - input_text_list (list): List of input texts.
    """

    # Define data directory and value pairs
    data_dir = f'./filtered_dataset_{dataset}/value_{num_value}/'
    value_pairs = list(itertools.combinations(value_dimensions, num_value))

    # Initialize an empty list to store input texts
    input_text_list = []

    # Iterate through each pair of value dimensions
    for pair_index, pair in enumerate(value_pairs):
        # Construct data path based on the number of value dimensions
        data_path = os.path.join(data_dir, f'{mode}_data', f'{pair_index}_{"_".join(pair)}_{mode}.json')
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Extract input texts from each sample in the data
        for sample in data:
            scenario = sample['scenario']
            for action in sample['actions']:
                action_desc = action['description']

                # Concatenate scenario and action description to form the input text
                input_text = scenario + " " + action_desc
                input_text_list.append(input_text)

    return input_text_list

def load_and_shuffle_data(data_dir, mode, value_pairs):
    all_samples = []

    for pair_index, pair in tqdm(enumerate(value_pairs), total=len(value_pairs)):
        data_path = os.path.join(data_dir, f'{mode}_data', f'{pair_index}_{"_".join(pair)}_{mode}.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
            for sample in data:
                sample['value_pair_index'] = pair_index
                all_samples.append(sample)

    random.shuffle(all_samples)
    return all_samples

def preprocess_sample_separate(sample, device='cuda', encoder_model_name='t5-base', target_seq_len = 100):
    device = device

    # Initialize tokenizer and T5 model
    if encoder_model_name == 't5-base':
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        encoder_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base", output_hidden_states=True).to(device)
    elif encoder_model_name == 't5-small':
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        encoder_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small", output_hidden_states=True).to(device)
    elif encoder_model_name == 'flan-t5-base':
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        encoder_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", output_hidden_states=True).to(device)
    elif encoder_model_name == 'bert-base-uncased':
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        encoder_model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased", output_hidden_states=True).to(device)
    elif encoder_model_name == 'roberta-base':
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        encoder_model = AutoModel.from_pretrained("FacebookAI/roberta-base", output_hidden_states=True).to(device)
    for param in encoder_model.parameters():
        param.requires_grad = False

    # Set up adaptive pooling layer
    target_seq_len = target_seq_len
    adaptive_pooling_layer = nn.AdaptiveAvgPool1d(target_seq_len)

    # Lists to store processed data for each action
    Scenario_Tensor, X_Data_list, y_Data_list, z_Data_list, action_list = [], [], [], [], []

    scenario = sample['scenario']
    input_text = scenario
    inputs = tokenizer(input_text, return_tensors="pt", max_length=2048, padding=True, truncation=True).to(device)
    
    if encoder_model.config.model_type == 't5':
        inputs["decoder_input_ids"] = inputs["input_ids"]
    with torch.no_grad():
        outputs = encoder_model(**inputs)

    # Get the last hidden state
    if encoder_model.config.model_type == 't5':
        last_hidden_state = outputs.encoder_last_hidden_state
    else:
        last_hidden_state = outputs.hidden_states[-1]
    pooled_output = adaptive_pooling_layer(last_hidden_state.permute(0, 2, 1)).permute(0, 2, 1)
    pooled_output = pooled_output.squeeze(0)

    Scenario_Tensor.append(pooled_output.unsqueeze(0))
    Scenario_Tensor = torch.cat(Scenario_Tensor, dim=0)

    # Iterate through each action in the sample
    for action in sample['actions']:
        action_desc = action['description']
        # action_values = torch.tensor(action['values'], dtype=torch.float32)

        # Prepare the input text and encode using T5
        input_text = action_desc
        inputs = tokenizer(input_text, return_tensors="pt", max_length=2048, padding=True, truncation=True).to(device)
        
        if encoder_model.config.model_type == 't5':
            inputs["decoder_input_ids"] = inputs["input_ids"]
        with torch.no_grad():
            outputs = encoder_model(**inputs)

        # Get the last hidden state
        if encoder_model.config.model_type == 't5':
            last_hidden_state = outputs.encoder_last_hidden_state
        else:
            last_hidden_state = outputs.hidden_states[-1]
            
        pooled_output = adaptive_pooling_layer(last_hidden_state.permute(0, 2, 1)).permute(0, 2, 1)
        pooled_output = pooled_output.squeeze(0)

        # Append the data to lists
        X_Data_list.append(pooled_output.unsqueeze(0))
        # y_Data_list.append(action_values.unsqueeze(0))
        # z_Data_list.append((action_values != 0).int().unsqueeze(0))
        action_list.append(action_desc)

    # Convert lists to tensors
    X_Data_tensor = torch.cat(X_Data_list, dim=0)
    # y_Data_tensor = torch.cat(y_Data_list, dim=0)
    # z_Data_tensor = torch.cat(z_Data_list, dim=0)

    # return Scenario_Tensor, X_Data_tensor, y_Data_tensor, z_Data_tensor, scenario, action_list
    return Scenario_Tensor, X_Data_tensor, scenario, action_list

def preprocess_chunk(device, chunk_samples, value_pairs, adaptive_pooling_layer, tokenizer, encoder_model):
    device = torch.device(device)

    # Lists to store processed data for the chunk
    X_Data = []
    y_Data = []
    z_Data = []
    input_text_list = []

    # Preprocess each sample in the chunk
    for sample in tqdm(chunk_samples, desc='Preprocessing chunk'):
        scenario = sample['scenario']
        pair_index = sample['value_pair_index']
        pair = value_pairs[pair_index]

        for action in sample['actions']:
            action_desc = action['description']
            action_values = torch.tensor(action['values'], dtype=torch.float32)

            # Prepare the input text and encode using encoder
            input_text = scenario + " " + action_desc
            inputs = tokenizer(input_text, return_tensors="pt", max_length=2048, padding=True, truncation=True).to(device)

            if encoder_model.config.model_type == 't5':
                inputs["decoder_input_ids"] = inputs["input_ids"]
            with torch.no_grad():
                outputs = encoder_model(**inputs)

            # Get the last hidden state
            if encoder_model.config.model_type == 't5':
                last_hidden_state = outputs.encoder_last_hidden_state
            else:
                last_hidden_state = outputs.hidden_states[-1]

            pooled_output = adaptive_pooling_layer(last_hidden_state.permute(0, 2, 1)).permute(0, 2, 1)
            pooled_output = pooled_output.squeeze(0)

            # Append the data to lists
            X_Data.append(pooled_output)
            y_Data.append(action_values)
            z_Data.append((action_values != 0).int())
            input_text_list.append(input_text)

    # Convert lists to tensors
    X_Data_tensor = torch.stack(X_Data)
    y_Data_tensor = torch.stack(y_Data)
    z_Data_tensor = torch.stack(z_Data)

    return X_Data_tensor, y_Data_tensor, z_Data_tensor, input_text_list

def preprocess_and_save_in_chunks(device='gpu:0', encoder_model_name='t5-base', num_values=[1], mode='test', value_dimensions=['curiosity', 'energy', 'safety', 'happiness', 'intimacy', 'fairness'], data_dir='./filtered_dataset_gpt4', save_dir='./', prefix='AllData', num_chunks=6):
    """
    Preprocesses the dataset in chunks and saves each chunk separately.

    Args:
    - device (str): Device identifier (gpu:0, cpu).
    - encoder_model_name (str): The name of encoder model.
    - num_value (int): Number of value dimensions to consider.
    - dataset (str): Dataset identifier.
    - mode (str): Mode of the dataset (train, test, validation).
    - value_dimensions (list): List of value dimensions.
    - base_path (str): Base directory to save the preprocessed data.
    - prefix (str): Prefix for the saved chunk files.
    - num_chunks (int): Number of chunks to split the data into.

    Returns:
    - None
    """
    device = torch.device(device)

    # Initialize tokenizer and encoder model
    if encoder_model_name == 't5-base':
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        encoder_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base", output_hidden_states=True).to(device)
    elif encoder_model_name == 't5-small':
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        encoder_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small", output_hidden_states=True).to(device)
    elif encoder_model_name == 'flan-t5-base':
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        encoder_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", output_hidden_states=True).to(device)
    elif encoder_model_name == 'bert-base-uncased':
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        encoder_model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased", output_hidden_states=True).to(device)
    elif encoder_model_name == 'roberta-base':
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        encoder_model = AutoModel.from_pretrained("FacebookAI/roberta-base", output_hidden_states=True).to(device)

    for param in encoder_model.parameters():
        param.requires_grad = False

    # Set up adaptive pooling layer
    target_seq_len = 100
    adaptive_pooling_layer = nn.AdaptiveAvgPool1d(target_seq_len)

    for num_value in num_values:
        # Define data directory and value pairs
        data_dir_new = f'{data_dir}value_{num_value}/'
        save_path = f'{save_dir}value_{num_value}/{mode}_data/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        value_pairs = list(itertools.combinations(value_dimensions, num_value))

        # Load and shuffle all data
        all_samples = load_and_shuffle_data(data_dir_new, mode, value_pairs)

        # Calculate chunk size
        total_samples = len(all_samples)
        chunk_size = total_samples // num_chunks

        # Process and save each chunk
        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = (chunk_id + 1) * chunk_size if chunk_id != num_chunks - 1 else total_samples

            chunk_samples = all_samples[start_idx:end_idx]

            # Preprocess the chunk
            X_chunk, y_chunk, z_chunk, _ = preprocess_chunk(device, chunk_samples, value_pairs, adaptive_pooling_layer, tokenizer, encoder_model)

            if num_chunks != 1:
                # Save the preprocessed chunk
                torch.save(X_chunk, os.path.join(save_path, f'X_{prefix}_{chunk_id+1}.pt'))
                torch.save(y_chunk, os.path.join(save_path, f'y_{prefix}_{chunk_id+1}.pt'))
                torch.save(z_chunk, os.path.join(save_path, f'z_{prefix}_{chunk_id+1}.pt'))
                print(f'Saved chunk {chunk_id+1} with size {X_chunk.size()}')
            else:
                torch.save(X_chunk, os.path.join(save_path, f'X_{prefix}.pt'))
                torch.save(y_chunk, os.path.join(save_path, f'y_{prefix}.pt'))
                torch.save(z_chunk, os.path.join(save_path, f'z_{prefix}.pt'))

if __name__ == '__main__':
    # Check if GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--encoder_model', type=str, default='t5-base', help='Encoder model to use', 
                        choices=['t5-small', 't5-base', 'flan-t5-base', 'bert-base-uncased', 'roberta-base'])
    parser.add_argument('-m', '--mode', type=str, default='test', help='Mode of the dataset', choices=['train', 'test'])
    parser.add_argument('-c', '--num_chunks', type=int, default=4, help='Number of chunks to split the data into')
    args = parser.parse_args()
    encoder_model_name = args.encoder_model
    mode = args.mode
    num_chunks = args.num_chunks
    if mode == 'test':
        num_chunks = 1

    num_values = [1,2,3,4,5,6]
    dataset = 'gpt4'
    data_dir = f'./filtered_dataset_{dataset}/'
    value_dimensions = ['curiosity', 'energy', 'safety', 'happiness', 'intimacy', 'fairness']

    # Directory to save the preprocessed data
    save_dir = f'./preprocessed_dataset/{encoder_model_name}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Preprocess and save data in chunks
    preprocess_and_save_in_chunks(device, encoder_model_name, num_values, mode, value_dimensions, data_dir, save_dir, 'AllData', num_chunks)