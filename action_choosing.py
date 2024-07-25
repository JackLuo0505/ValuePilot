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

# Setting seed for reproducibility
set_seed(42)

def calculate_scores_weighted_sum(sample, preferences):
    """
    Calculate the weighted sum of scores for each action in the sample
    :param sample: Sample containing ratings and preferences
    :param preferences: Personal preference vector
    :return: Weighted sum score for each action
    """
    predicted_values_rating = np.array(sample['predicted_values_rating'])
    
    # 计算加权和
    weighted_sum = np.dot(preferences, predicted_values_rating)
    
    return weighted_sum

def sort_actions(results, preferences):
    """
    Sort actions based on their weighted sum scores
    :param results: Results containing multiple samples
    :param preferences: Personal preference vector
    :return: Sorted list of actions, each containing weighted sum score, index, rating, and actual values
    """
    action_list = []

    for index, sample in enumerate(results):
        score = calculate_scores_weighted_sum(sample, preferences)
        action_info = {
            'score': score,
            'index': index,
            'rating': sample['predicted_values_rating'],
            'actual_values': sample['actual_values_rating']
        }
        action_list.append(action_info)

    # 按 normalized score 对动作进行排序
    sorted_actions = sorted(action_list, key=lambda x: x['score'], reverse=True)

    return sorted_actions

def sort_actions_no_actual_values(results, preferences):
    """
    Sort actions based on their weighted sum scores (without actual values)
    :param results: Results containing multiple samples
    :param preferences: Personal preference vector
    :return: Sorted list of actions, each containing weighted sum score, index, and rating
    """
    action_list = []

    for index, sample in enumerate(results):
        score = calculate_scores_weighted_sum(sample, preferences)
        action_info = {
            'score': score,
            'index': index,
            'rating': sample['predicted_values_rating'],
        }
        action_list.append(action_info)

    # 按 normalized score 对动作进行排序
    sorted_actions = sorted(action_list, key=lambda x: x['score'], reverse=True)

    return sorted_actions

def sort_actions_separate_no_actual_values(results, preferences):
    """
    Sort actions based on their weighted sum scores, considering separate predicted ratings (without actual values)
    :param results: Results containing multiple samples
    :param preferences: Personal preference vector
    :return: Sorted list of actions, each containing weighted sum score, index, and separate predicted ratings
    """
    action_list = []

    for index, sample in enumerate(results):
        scenario_predictied_rating = np.array(sample['scenario_predictied_rating']) * 0.3
        predicted_values_rating = np.array(sample['predicted_values_rating']) * 0.7
        rating = scenario_predictied_rating + predicted_values_rating

        score = np.dot(preferences, rating)

        action_info = {
            'score': score,
            'index': index,
            'scenario predictied rating': scenario_predictied_rating,
            'predicted values rating': predicted_values_rating,
            'rating': rating,
        }
        action_list.append(action_info)

    # 按 normalized score 对动作进行排序
    sorted_actions = sorted(action_list, key=lambda x: x['score'], reverse=True)

    return sorted_actions

def promethee_sort_actions(results, preferences):
    """
    Use PROMETHEE sorting method to rank actions
    :param results: Results containing multiple samples
    :param preferences: Personal preference vector
    :return: Sorted list of actions, each containing net flow, index, and ratings
    """
    num_actions = len(results)
    ratings = []
    
    # Normalize weights
    preferences = preferences / np.sum(preferences)

    # Calculate overall ratings for each action
    for sample in results:
        scenario_predicted_rating = np.array(sample['scenario_predictied_rating']) * 0.3
        predicted_values_rating = np.array(sample['predicted_values_rating']) * 0.7
        rating = scenario_predicted_rating + predicted_values_rating
        ratings.append(rating)
    ratings = np.array(ratings)

    # Define preference function
    def sigmoid_preference_function(option_a_score, option_b_score, k=1):
        diff = option_a_score - option_b_score
        return 1 / (1 + np.exp(-k * diff))

    # Calculate preference matrix
    preference_matrix = np.zeros((num_actions, num_actions, len(rating)))
    for i in range(num_actions):
        for j in range(num_actions):
            for k in range(len(rating)):
                preference_matrix[i, j, k] = sigmoid_preference_function(ratings[i, k], ratings[j, k])

    # Calculate weighted preferences using weighted product method
    weighted_preferences = np.ones((num_actions, num_actions))
    for k in range(len(preferences)):
        weighted_preferences *= preference_matrix[:, :, k] ** preferences[k]

    # Calculate positive and negative flows
    positive_flow = np.mean(weighted_preferences, axis=1)
    negative_flow = np.mean(weighted_preferences, axis=0)

    # Calculate net flow
    net_flow = positive_flow - negative_flow

    # Create action list and attach score information
    sorted_actions = []
    for index, sample in enumerate(results):
        action_info = {
            'score': net_flow[index],  # Use net flow as the score
            'index': index,
            'scenario predictied rating': np.array(sample['scenario_predictied_rating']) * 0.3,
            'predicted values rating': np.array(sample['predicted_values_rating']) * 0.7,
            'rating': ratings[index],
        }
        sorted_actions.append(action_info)

    # Sort actions by net flow (score)
    sorted_actions = sorted(sorted_actions, key=lambda x: x['score'], reverse=True)

    return sorted_actions

def promethee_sort_actions_close_to_preferences(results, preferences):
    """
    Use PROMETHEE sorting method to rank actions based on closeness to preferences
    :param results: Results containing multiple samples
    :param preferences: Personal preference vector
    :return: Sorted list of actions, each containing net flow, index, and ratings
    """
    num_actions = len(results)
    predicted_values_ratings = []
    scenario_predicted_ratings = []
    ratings = []
    
    # Calculate overall ratings for each action
    for sample in results:
        scenario_predicted_rating = np.array(sample['scenario_predictied_rating'])
        predicted_values_rating = np.array(sample['predicted_values_rating'])
        scenario_predicted_rating = 1 - np.abs(scenario_predicted_rating - preferences)
        predicted_values_rating = 1 - np.abs(predicted_values_rating - preferences)
        scenario_predicted_rating = scenario_predicted_rating * 0.3
        predicted_values_rating = predicted_values_rating * 0.7
        rating = scenario_predicted_rating + predicted_values_rating
        scenario_predicted_ratings.append(scenario_predicted_rating)
        predicted_values_ratings.append(predicted_values_rating)
        ratings.append(rating)
    ratings = np.array(ratings)

    # Normalize weights
    preferences = preferences / np.sum(preferences)

    # 定义偏好函数
    def sigmoid_preference_function(option_a_score, option_b_score, k=1):
        diff = option_a_score - option_b_score
        return 1 / (1 + np.exp(-k * diff))

    # Define preference function
    def sigmoid_preference_function(option_a_score, option_b_score, k=1):
        diff = option_a_score - option_b_score
        return 1 / (1 + np.exp(-k * diff))

    # Calculate preference matrix
    preference_matrix = np.zeros((num_actions, num_actions, len(rating)))
    for i in range(num_actions):
        for j in range(num_actions):
            for k in range(len(rating)):
                preference_matrix[i, j, k] = sigmoid_preference_function(ratings[i, k], ratings[j, k])

    # Calculate weighted preferences using weighted product method
    weighted_preferences = np.ones((num_actions, num_actions))
    for k in range(len(preferences)):
        weighted_preferences *= preference_matrix[:, :, k] ** preferences[k]

    # Calculate positive and negative flows
    positive_flow = np.mean(weighted_preferences, axis=1)
    negative_flow = np.mean(weighted_preferences, axis=0)

    # Calculate net flow
    net_flow = positive_flow - negative_flow

    # Create action list and attach score information
    sorted_actions = []
    for index, sample in enumerate(results):
        action_info = {
            'score': net_flow[index],  # Use net flow as the score
            'index': index,
            'scenario predictied rating': scenario_predicted_ratings[index],
            'predicted values rating': predicted_values_ratings[index],
            'rating': ratings[index],
        }
        sorted_actions.append(action_info)

    # Sort actions by net flow (score)
    sorted_actions = sorted(sorted_actions, key=lambda x: x['score'], reverse=True)

    return sorted_actions

def promethee_sort_actions_synthesis(results, preferences):
    """
    Use PROMETHEE sorting method to rank actions with synthesis of predicted ratings and preferences
    :param results: Results containing multiple samples
    :param preferences: Personal preference vector
    :return: Sorted list of actions, each containing net flow, index, and ratings
    """
    num_actions = len(results)
    predicted_values_ratings = []
    scenario_predicted_ratings = []
    ratings = []

    def correcting_preference(x):
        return 1 / (1 + np.exp(-(x - 0.5 ) * 10))
    
    preferences = correcting_preference(preferences)
    
    # Calculate overall ratings for each action
    for sample in results:
        scenario_predicted_rating = np.array(sample['scenario_predictied_rating'])
        predicted_values_rating = np.array(sample['predicted_values_rating'])
        scenario_predicted_rating = correcting_preference(scenario_predicted_rating)
        predicted_values_rating = correcting_preference(predicted_values_rating)
        scenario_predicted_rating = (1 - np.abs(np.abs(scenario_predicted_rating) - preferences)) * 0.3
        predicted_values_rating = (1 - np.abs(np.abs(predicted_values_rating) - preferences)) * 0.3
        scenario_predicted_rating = scenario_predicted_rating + np.array(sample['scenario_predictied_rating']) * 0.7
        predicted_values_rating = predicted_values_rating + np.array(sample['predicted_values_rating']) * 0.7
        rating = 1 / (1 + np.exp( - np.abs(scenario_predicted_rating))) * predicted_values_rating
        
        scenario_predicted_ratings.append(scenario_predicted_rating)
        predicted_values_ratings.append(predicted_values_rating)
        rating = rating / np.sum(rating)
        ratings.append(rating)
    ratings = np.array(ratings)

    # Normalize weights
    preferences = preferences / np.sum(preferences)

    # Define preference function
    def sigmoid_preference_function(option_a_score, option_b_score, k=1):
        diff = option_a_score - option_b_score
        return 1 / (1 + np.exp(-k * diff))

    # Calculate preference matrix
    preference_matrix = np.zeros((num_actions, num_actions, len(rating)))
    for i in range(num_actions):
        for j in range(num_actions):
            for k in range(len(rating)):
                preference_matrix[i, j, k] = sigmoid_preference_function(ratings[i, k], ratings[j, k])

    # Calculate weighted preferences using weighted product method
    weighted_preferences = np.ones((num_actions, num_actions))
    for k in range(len(preferences)):
        weighted_preferences += preference_matrix[:, :, k] * preferences[k]

    # Calculate positive and negative flows
    positive_flow = np.mean(weighted_preferences, axis=1)
    negative_flow = np.mean(weighted_preferences, axis=0)

    # Calculate net flow
    net_flow = positive_flow - negative_flow

    # Create action list and attach score information
    sorted_actions = []
    for index, sample in enumerate(results):
        action_info = {
            'score': net_flow[index],  # Use net flow as the score
            'index': index,
            'scenario predictied rating': scenario_predicted_ratings[index],
            'predicted values rating': predicted_values_ratings[index],
            'rating': ratings[index],
        }
        sorted_actions.append(action_info)

    # Sort actions by net flow (score)
    sorted_actions = sorted(sorted_actions, key=lambda x: x['score'], reverse=True)

    return sorted_actions

def visualize_normalized_scores(normalized_score_list, output_folder = "action_chosing_result"):
    """
    Visualize normalized_score_list using a scatter plot, with the maximum value highlighted in red
    """
    max_index = normalized_score_list.index(max(normalized_score_list))

    plt.scatter(range(len(normalized_score_list)), normalized_score_list, label='Normalized Scores')
    plt.scatter(max_index, normalized_score_list[max_index], color='red', label='Max Value')  # Highlight max value in red
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Score')
    plt.title('Normalized Scores for Samples')
    plt.legend()

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save plot to file
    output_path = os.path.join(output_folder, 'normalized_scores_scatter.png')
    plt.savefig(output_path)
    plt.close()

def remove_negative_zero(value):
    return 0 if abs(value) == 0 else value

def print_actions_details(preferences, scenario, action_list, sorted_actions):
    """
    Print detailed information of actions in a table format
    """

    # Print formatted scenario
    print("\nScenario:")
    formatted_scenario = textwrap.fill(scenario, width=85)
    print(formatted_scenario)
    
    for i, action_info in enumerate(sorted_actions):
        action_index = action_info['index']
        action = action_list[action_index]
        action_actual_values = action_info['actual_values']
        action_rating = action_info['rating']
        score = action_info['score']

        table = PrettyTable()
        table.title = f'Details for Action {i + 1}'
        table.field_names = ['Value', 'Personal Preferences', 'Action Actual Values', 'Action Rating']
        for label, pref, actual, rating in zip(['Curiosity', 'Energy', 'Safety', 'Happiness', 'Intimacy', 'Fairness'],
                                               preferences, action_actual_values, action_rating):
            pref = remove_negative_zero(round(pref, 3))
            actual = remove_negative_zero(round(actual, 3))
            rating = remove_negative_zero(round(rating, 3))
            table.add_row([label, f'{pref:.3f}', f'{actual:.3f}', f'{rating:.3f}'])

        print(f"Action {i + 1}:")
        print(f"{action}")
        print(f"The Score of Action {i + 1}:")
        print(f"{score}")

        # Print table
        print(table)
        print("")
    
    # Print summary table
    print("\nSummary Table:")
    summary_table = PrettyTable()
    summary_table.field_names = ['Action Index', 'Action', 'Score']
    for action_info in sorted_actions:
        action_index = action_info['index']
        action_description = action_list[action_index]
        score = action_info['score']
        action_description_wrapped = "\n".join([action_description[i:i+50] for i in range(0, len(action_description), 50)])
        summary_table.add_row([action_index + 1, action_description_wrapped, f'{score:.3f}'])
    print(summary_table)

def print_actions_details_no_actual_values(preferences, scenario, action_list, sorted_actions):
    """
    Print detailed information of actions in a table format (without actual values)
    """

    # Print formatted scenario
    print("\nScenario:")
    formatted_scenario = textwrap.fill(scenario, width=85)
    print(formatted_scenario)
    
    for i, action_info in enumerate(sorted_actions):
        action_index = action_info['index']
        action = action_list[action_index]
        action_rating = action_info['rating']
        score = action_info['score']

        table = PrettyTable()
        table.title = f'Details for Action {i + 1}'
        table.field_names = ['Value', 'Personal Preferences', 'Action Rating']
        for label, pref, rating in zip(['Curiosity', 'Energy', 'Safety', 'Happiness', 'Intimacy', 'Fairness'],
                                               preferences, action_rating):
            pref = remove_negative_zero(round(pref, 3))
            rating = remove_negative_zero(round(rating, 3))
            table.add_row([label, f'{pref:.3f}', f'{rating:.3f}'])

        print(f"Action {i + 1}:")
        print(f"{action}")
        print(f"The Score of Action {i + 1}:")
        print(f"{score}")

        # Print table
        print(table)
        print("")
    
    # Print summary table
    print("\nSummary Table:")
    summary_table = PrettyTable()
    summary_table.field_names = ['Action Index', 'Action', 'Score']
    for action_info in sorted_actions:
        action_index = action_info['index']
        action_description = action_list[action_index]
        score = action_info['score']
        action_description_wrapped = "\n".join([action_description[i:i+100] for i in range(0, len(action_description), 50)])
        summary_table.add_row([action_index + 1, action_description_wrapped, f'{score:.3f}'])
    print(summary_table)

def save_actions_details_no_actual_values(preferences, scenario, action_list, sorted_actions, md_filename, order_json_filename, i):
    """
    Save detailed information of actions in Markdown format and save the results to a Markdown file.
    """

    if not os.path.exists(order_json_filename):
        with open(order_json_filename, 'w') as json_file:
            json.dump([], json_file)

    with open(md_filename, 'a') as md_file:
        # Write scenario description
        md_file.write(f"#### question {i}:\n")
        md_file.write("##### Scenario:\n")
        md_file.write(scenario + "\n\n")

        # Iterate through sorted actions and write
        for idx, action_info in enumerate(sorted_actions):
            action_index = action_info['index']
            action = action_list[action_index]
            action_rating = action_info['rating']
            score = action_info['score']

            md_file.write(f"###### Action {idx + 1}:")
            md_file.write(f" {action_index + 1}\n")
            md_file.write(action + "\n\n")
            md_file.write(f"The Score of Action {idx + 1}: ")
            md_file.write(f"{score}\n\n")

            md_file.write("| Value | Personal Preferences | Action Rating |\n")
            md_file.write("|-------|----------------------|---------------|\n")
            for label, pref, rating in zip(['Curiosity', 'Energy', 'Safety', 'Happiness', 'Intimacy', 'Fairness'],
                                           preferences, action_rating):
                pref = remove_negative_zero(round(pref, 3))
                rating = remove_negative_zero(round(rating, 3))
                md_file.write(f"| {label} | {pref:.3f} | {rating:.3f} |\n")

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
        
        with open(order_json_filename, 'r') as json_file:
            data = json.load(json_file)
        action_index_order = [action_info['index'] + 1 for action_info in sorted_actions]
        new_data = {
            "question": i,
            "model output order": action_index_order,
            "tester order": []
        }
        data.append(new_data)
        with open(order_json_filename, 'w') as json_file:
            json.dump(data, json_file, indent = 2)

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
    

def visualize_radar_chart(preferences, actual_values, ratings, title, labels=['Curiosity', 'Energy', 'Safety', 'Happiness', 'Intimacy', 'Fairness'], output_folder="action_chosing_result"):
    """
    Visualize multiple datasets using radar chart and save to the specified folder
    """
    center = -1
    outer = 1
    
    # Angle settings
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    # Calculate normalized values between center and outer
    normalized_preferences = [(value - center) / (outer - center) for value in preferences]
    normalized_actual_values = [(value - center) / (outer - center) for value in actual_values]
    normalized_ratings = [(value - center) / (outer - center) for value in ratings]

    # Plot radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, normalized_preferences, color='red', alpha=0.25, label='Preferences')
    ax.fill(angles, normalized_actual_values, color='blue', alpha=0.25, label='Actual Values')
    ax.fill(angles, normalized_ratings, color='green', alpha=0.25, label='Rating Values')

    # Set labels
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)

    # Display values for each point
    for i, angle in enumerate(angles):
        ax.text(angle, normalized_preferences[i], '{:.2f}'.format(normalized_preferences[i]), color='red', ha='left', va='bottom')
        ax.text(angle, normalized_actual_values[i], '{:.2f}'.format(normalized_actual_values[i]), color='blue', ha='left', va='top')
        ax.text(angle, normalized_ratings[i], '{:.2f}'.format(normalized_ratings[i]), color='green', ha='right', va='top')

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save plot
    output_path = os.path.join(output_folder, 'radar_chart.png')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Test Script for Evaluating Rating and Labeling Models')
    parser.add_argument('-e', '--encoder_model', type=str, default='t5-base', help='Encoder model used',
                        choices=['t5-small', 't5-base', 'flan-t5-base', 'bert-base-uncased', 'roberta-base'])
    encoder_model = parser.parse_args().encoder_model
    
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
            sorted_actions = promethee_sort_actions_synthesis(result, preferences)

            # Print all sorted actions information
            save_actions_details_separate_no_actual_values(preferences, scenario, action_list, sorted_actions, f"{save_pth}test_questionnaire_{tester+1}.md", f"{save_pth}order_record_{tester+1}.json", i + 1)