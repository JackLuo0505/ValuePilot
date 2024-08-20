"""
action_choosing.py
Author: Anonymous Author
Date: ---

This script provides various functions to evaluate and rank actions based on user preferences using different methods, including weighted sum and PROMETHEE sorting. It integrates results from models trained with Adaptive Complexity Curriculum Learning (ACCL) and is used for sequential evaluation in different scenarios.

In this script:
- Functions to calculate weighted scores for actions.
- Functions to sort actions based on different criteria.
- Visualization tools to present the results.
- Integration with pre-trained LabelNet (LN) and RateNet (RN) models for action evaluation.
"""
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import textwrap
import json
from prettytable import PrettyTable
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from networks import LabelNet, RateNet
from utils import set_seed, load_model_and_optimizer, get_specified_order_name
from inference_combine_results import test_and_return_result, test_and_return_combined_result_separate
from preprocess import preprocess_sample_separate
from MCDM import promethee_sort_actions_synthesis, ahp_sort_actions_synthesis, topsis_sort_actions, maut_sort_actions

# Setting seed for reproducibility
set_seed(42)


def remove_negative_zero(value):
    return 0 if abs(value) == 0 else value

def save_actions_details_separate_no_actual_values(preferences, scenario, action_list, sorted_actions, md_filename, order_json_filename, i):
    """
    Save detailed information of actions in Markdown format with separate ratings and save the results to a Markdown file.
    """

    # Ensure the directories exist
    md_dir = os.path.dirname(md_filename)
    json_dir = os.path.dirname(order_json_filename)
    
    os.makedirs(md_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    if not os.path.exists(order_json_filename):
        with open(order_json_filename, 'w') as json_file:
            json.dump([], json_file)

    with open(md_filename, 'a') as md_file:
        # Write scenario description
        md_file.write(f"#### question {i}:\n")
        md_file.write("##### Scenario:\n")
        md_file.write(scenario + "\n\n")

        # Iterate through sorted actions and write detailed information to the Markdown file
        for idx, action_info in enumerate(sorted_actions):
            action_index = action_info['index']
            action = action_list[action_index]
            predicted_values_rating = action_info['predicted values rating']
            scenario_predictied_rating = action_info['scenario predictied rating']
            ratings = action_info['rating']
            score = action_info['score']

            md_file.write(f"###### Action {idx + 1}:")
            md_file.write(f" {action_index + 1}\n")
            md_file.write(action + "\n\n")
            md_file.write(f"The Score of Action {idx + 1}: ")
            md_file.write(f"{score}\n\n")

            md_file.write("| Value | Personal Preferences | Scenario Rating | Action Rating | Total Rating |\n")
            md_file.write("|-------|----------------------|-----------------|---------------|--------------|\n")
            for label, pref, values_rating, scenario_rating, rating in zip(['Curiosity', 'Energy', 'Safety', 'Happiness', 'Intimacy', 'Fairness'],
                                           preferences, predicted_values_rating, scenario_predictied_rating, ratings):
                pref = remove_negative_zero(round(pref, 3))
                scenario_rating = remove_negative_zero(round(scenario_rating, 3))
                values_rating = remove_negative_zero(round(values_rating, 3))
                rating = remove_negative_zero(round(rating, 3))
                md_file.write(f"| {label} | {pref:.3f} | {scenario_rating:.3f} | {values_rating:.3f} | {rating:.3f} |\n")

            md_file.write("\n\n")

        # Write summary table
        md_file.write("###### Summary Table:\n")
        md_file.write("| Action Index | Action | Score |\n")
        md_file.write("|--------------|--------|-------|\n")
        for action_info in sorted_actions:
            action_index = action_info['index']
            action_description = action_list[action_index]
            score = action_info['score']
            md_file.write(f"| {action_index + 1} | {action_description} | {score:.3f} |\n")
        
        try:
            with open(order_json_filename, 'r') as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError:
            data = []
        
        action_index_order = [action_info['index'] + 1 for action_info in sorted_actions]
        new_data = {
            "question": i,
            "model output order": action_index_order,
            "tester order": []
        }
        data.append(new_data)
        with open(order_json_filename, 'w') as json_file:
            json.dump(data, json_file, indent = 2)


if __name__ == "__main__":
    # Setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Test Script for Evaluating Rating and Labeling Models')
    parser.add_argument('-e', '--encoder_model', type=str, default='t5-base', help='Encoder model used',
                        choices=['t5-small', 't5-base', 'flan-t5-base', 'bert-base-uncased', 'roberta-base'])
    parser.add_argument('-m', '--MCDM_method', type=str, default='promethee', help='MCDM method used',
                        choices=['promethee', 'ahp', 'topsis', 'maut'])
    encoder_model = parser.parse_args().encoder_model
    MCDM_method = parser.parse_args().MCDM_method
    
    # Model configuration
    labeling_model_type = "ACCL"
    rating_model_type = "ACCL_CJRN"
    data_mode = "gpt4"
    labeling_learning_rate = 1e-3
    labeling_weight_decay = 1e-5
    threshold = 0.8
    rating_learning_rate = 1e-5
    rating_weight_decay = 1e-4
    model_included_value = get_specified_order_name([6,1,5,2,4,3])

    # labeling_model_name = f"{labeling_model_type}_{data_mode}_avg_128_{labeling_learning_rate}_{labeling_weight_decay}_{threshold}"
    # rating_model_name = f"{rating_model_type}_{data_mode}_avg_128_{rating_learning_rate}_{rating_weight_decay}"
    
    # Loading trained models
    if encoder_model == 't5-small':
        hidden_size = 512
    else:
        hidden_size = 768
    loaded_label_model = LabelNet(hidden_size, 4, 128).to(device)
    loaded_label_optimizer = torch.optim.Adam(loaded_label_model.parameters(), lr=labeling_learning_rate, weight_decay=labeling_weight_decay)
    label_model, label_optimizer = load_model_and_optimizer(loaded_label_model, loaded_label_optimizer,
                                                             f'./model_save/{encoder_model}/value_{model_included_value}/label_model.pth',
                                                             f'./model_save/{encoder_model}/value_{model_included_value}/label_optimizer.pth')

    loaded_rate_model = RateNet(hidden_size, 4, 128).to(device)
    loaded_rate_optimizer = torch.optim.Adam(loaded_rate_model.parameters(), lr=rating_learning_rate, weight_decay=rating_weight_decay)
    rate_model, rate_optimizer = load_model_and_optimizer(loaded_rate_model, loaded_rate_optimizer,
                                                           f'./model_save/{encoder_model}/value_{model_included_value}/rate_model.pth',
                                                           f'./model_save/{encoder_model}/value_{model_included_value}/rate_optimizer.pth')

    
    data_path = f'./human_study/Test_questionnaire.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
        data = data['questions']
        
    preferences_file_path = f'./human_study/result_en.json'
    with open(preferences_file_path, 'r') as f:
        preferences_file = json.load(f)

    save_pth = f"./human_study/ValuePilot/{encoder_model}/"
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)

    for tester, tester_file in tqdm(enumerate(preferences_file)):
        preferences = np.array(tester_file['value_point'])
        for i, sample in enumerate(data):
            scenario_input, input_test, scenario, action_list = preprocess_sample_separate(sample, device=device, encoder_model_name=encoder_model)

            correct_rating_threshold = 0.2
            output_type = "Combine rating and labeling"
            if (output_type == "NOT Combine rating and labeling"):
                result, correct_num_list, all_correct, label_all_correct = test_and_return_result(rate_model, label_model, input_test, None, None, correct_rating_threshold, action_list)
            else:
                result, correct_num_list, all_correct, label_all_correct = test_and_return_combined_result_separate(rate_model, label_model, scenario_input, input_test, None, None, correct_rating_threshold, action_list)

            # Call function and get results
            if MCDM_method == 'promethee':
                sorted_actions = promethee_sort_actions_synthesis(result, preferences)
                # Print all sorted actions information
                save_actions_details_separate_no_actual_values(preferences, scenario, action_list, sorted_actions, f"{save_pth}test_questionnaire_{tester+1}.md", f"{save_pth}order_record_{tester+1}.json", i + 1)
            elif MCDM_method == 'ahp':
                sorted_actions = ahp_sort_actions_synthesis(result, preferences)
                # Print all sorted actions information
                save_actions_details_separate_no_actual_values(preferences, scenario, action_list, sorted_actions, f"human_study/Ablation_Experiment_MCDM/{MCDM_method}/test_questionnaire_{tester+1}.md", f"human_study/Ablation_Experiment_MCDM/{MCDM_method}/order_record_{tester+1}.json", i + 1)
            elif MCDM_method == 'topsis':
                sorted_actions = topsis_sort_actions(result, preferences)
                # Print all sorted actions information
                save_actions_details_separate_no_actual_values(preferences, scenario, action_list, sorted_actions, f"human_study/Ablation_Experiment_MCDM/{MCDM_method}/test_questionnaire_{tester+1}.md", f"human_study/Ablation_Experiment_MCDM/{MCDM_method}/order_record_{tester+1}.json", i + 1)
            elif MCDM_method == 'maut':
                sorted_actions = maut_sort_actions(result, preferences)
                # Print all sorted actions information
                save_actions_details_separate_no_actual_values(preferences, scenario, action_list, sorted_actions, f"human_study/Ablation_Experiment_MCDM/{MCDM_method}/test_questionnaire_{tester+1}.md", f"human_study/Ablation_Experiment_MCDM/{MCDM_method}/order_record_{tester+1}.json", i + 1)