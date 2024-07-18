"""
find_error_data.py
Author: Anonymous Author
Date: ---

This script processes JSON files containing scenario and action data to identify and filter out entries influenced by specific value dimensions. The value dimensions are predefined and include curiosity, energy, safety, happiness, intimacy, and fairness. For each value dimension, corresponding word forms are also considered.

In this script:
- Value dimensions and their corresponding word forms are defined.
- Value pairs are generated using combinations of the value dimensions.
- The script reads JSON files containing scenarios and actions.
- It checks whether scenarios or actions contain any of the value dimensions or their word forms.
- Entries influenced by the value dimensions are filtered out.
- The filtered data is written to new JSON files.
- The script provides a summary of the total number of matches and the proportion of actions influenced by value dimensions.
"""
import json
import itertools
from tqdm import tqdm
import os

# Define value dimensions
num_value = 6
dataset = 'gpt4'
data_dir = f'./dataset_{dataset}/value_{num_value}/'
mode = 'test'
output_dir = f'./filtered_dataset_{dataset}/value_{num_value}/{mode}_data/'
value_dimensions = ['curiosity', 'energy', 'safety', 'happiness', 'intimacy', 'fairness']

# Define word forms mapping
word_forms = {
    'curiosity': ['curious', 'curiously'],
    'energy': ['energetic', 'energetically'],
    'safety': ['safe', 'safely'],
    'happiness': ['happy', 'happily'],
    'intimacy': ['intimate', 'intimately'],
    'fairness': ['fair', 'fairly']
}

# Generate value pairs
all_value_pairs = list(itertools.combinations(value_dimensions, num_value))

# Initialize total counters
total_scenario_count = 0
total_action_count = 0
total_actions_influenced_by_value_pairs = 0
total_actions = 0

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read and process JSON files
for pair_index, pair in tqdm(enumerate(all_value_pairs), total=len(all_value_pairs)):
    data_path = os.path.join(data_dir, f'{mode}_data', f'{pair_index}_{"_".join(pair)}_{mode}.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    filtered_data = []
    scenario_count = 0
    action_count = 0
    actions_influenced_by_value_pairs = 0

    for entry in data:
        total_actions += len(entry['actions'])
        
        # Check if scenario contains any value pair or its word forms
        scenario = entry['scenario']
        scenario_influenced = False
        for value in pair:
            words_to_check = [value] + word_forms.get(value, [])
            for word in words_to_check:
                if word in scenario:
                    scenario_count += 1
                    scenario_influenced = True
                    actions_influenced_by_value_pairs += len(entry['actions'])
                    break
            if scenario_influenced:
                break
        
        if scenario_influenced:
            continue  # Skip this entry entirely

        # Process actions
        new_actions = []
        for action in entry['actions']:
            description = action['description']
            action_influenced = False
            for value in pair:
                words_to_check = [value] + word_forms.get(value, [])
                for word in words_to_check:
                    if word in description:
                        action_count += 1
                        actions_influenced_by_value_pairs += 1
                        action_influenced = True
                        break
                if action_influenced:
                    break
            if not action_influenced:
                new_actions.append(action)
        
        if new_actions:
            entry['actions'] = new_actions
            filtered_data.append(entry)

    # Update total counters
    total_scenario_count += scenario_count
    total_action_count += action_count
    total_actions_influenced_by_value_pairs += actions_influenced_by_value_pairs

    # Write filtered data to new JSON file
    output_path = os.path.join(output_dir, f'{pair_index}_{"_".join(pair)}_{mode}.json')
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    # Print results
    print(f"Pair: {pair}")
    print(f"Total entries containing value pairs or their word forms: {scenario_count + action_count}")
    print(f"Total matches in scenarios: {scenario_count}")
    print(f"Total matches in actions: {action_count}")
    print(f"Total actions influenced by value pairs or their word forms: {actions_influenced_by_value_pairs}")

# Summary results
print("\nSummary results:")
print(f"Total matches in scenarios: {total_scenario_count}")
print(f"Total matches in actions: {total_action_count}")
print(f"Total actions influenced by value pairs or their word forms: {total_actions_influenced_by_value_pairs}")
print(f"Total number of actions: {total_actions}")
if total_actions > 0:
    influenced_ratio = total_actions_influenced_by_value_pairs / total_actions
    print(f"Proportion of actions influenced by value pairs or their word forms: {influenced_ratio:.2%}")
else:
    print("No actions found")