"""
Test Script for Evaluating Rating and Labeling Models
Author: Anonymous Author
Date: ---

This script is used to evaluate the performance of rating and labeling models on a test dataset. The evaluation includes computing accuracy metrics and generating sample results for inspection.

In this script:
- Test data is loaded and preprocessed.
- Trained rating and labeling models are loaded from saved checkpoints.
- The test dataset is passed through the models, and performance metrics are computed.
- Results are printed, including accuracy metrics and sample predictions.
"""

# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from networks import LabelNet, RateNet
import argparse
from preprocess import get_input_text_list
from utils import set_seed, load_data_tensors, load_model_and_optimizer, compare_lists, normalize_list, convert_to_label, random_select_n_samples, print_accuracy, print_and_save_results, get_specified_order_name

# Setting seed for reproducibility
set_seed(42)

def test_and_return_result(model1, model2, test_input, test_target1, test_target2, threshold, input_text_list):
    """
    Function to test the rating and labeling models individually and return results.

    Args:
    - model1: The trained rating model.
    - model2: The trained labeling model.
    - test_input: Input data for testing.
    - test_target1: Target data for rating prediction.
    - test_target2: Target data for labeling prediction.
    - threshold: Threshold for correct rating predictions.
    - input_text_list: List of input text descriptions.

    Returns:
    - data_for_comparison: List of dictionaries containing sample predictions and actual values.
    - correct_num_list: List containing the number of correct predictions for each dimension.
    - all_correct: Total number of samples with all correct predictions.
    """
    device = next(model1.parameters()).device
    model1.eval()
    model2.eval()
    data_for_comparison = []
    correct_num_list = [0, 0, 0, 0, 0, 0]
    all_correct = 0
    label_all_correct = 0

    criterion = nn.MSELoss()

    # Move input and targets to the correct device
    test_input = test_input.to(device)
    if test_target1 is not None and test_target2 is not None:
        test_target1, test_target2 = test_target1.to(device), test_target2.to(device)

    with torch.no_grad():
        test_output1 = model1(test_input)
        test_output2 = model2(test_input)
        if test_target1 is None and test_target2 is None:
            test_target1 = torch.zeros_like(test_output1).to(device)
            test_target2 = torch.zeros_like(test_output2).to(device)

        for i in range(len(test_input)):
            test_input_sample = test_input[i]
            test_target_sample1 = test_target1[i]
            test_target_sample2 = test_target2[i]
            test_output_sample1 = test_output1[i]
            test_output_sample2 = test_output2[i]
            test_loss = criterion(test_output1[i], test_target1[i])
            output1_loss = test_loss.item()

            normalized_label = normalize_list(test_output_sample2)

            bool_list = compare_lists(test_output_sample1, test_target_sample1, threshold)
            label_bool_list = compare_lists(convert_to_label(normalized_label), test_target_sample2, 0.1)
            correct_num_list = [x + 1 if bool_val else x for x, bool_val in zip(correct_num_list, bool_list)]
            if all(bool_list):
                all_correct += 1
            if all(label_bool_list):
                label_all_correct += 1

            current_data = {
                'scenario_and_description': input_text_list[i],
                'initial_predicted_values_rating': test_output_sample1.cpu().detach().numpy().tolist(),
                'predicted_values_rating': test_output_sample1.cpu().detach().numpy().tolist(),
                'actual_values_rating': test_target_sample1.cpu().detach().numpy().tolist(),
                'output1_loss': output1_loss,
                'initial_predicted_values_labeling': test_output_sample2.cpu().detach().numpy().tolist(),
                'predicted_values_labeling': test_output_sample2.cpu().detach().numpy().tolist(),
                'actual_values_labeling': test_target_sample2.cpu().detach().numpy().tolist(),
            }
            data_for_comparison.append(current_data)

    return data_for_comparison, correct_num_list, all_correct, label_all_correct

def test_and_return_combined_result(model1, model2, test_input, test_target1, test_target2, threshold, input_text_list):
    """
    Function to test the combined rating and labeling models and return results.

    Args:
    - model1: The trained rating model.
    - model2: The trained labeling model.
    - test_input: Input data for testing.
    - test_target1: Target data for rating prediction.
    - test_target2: Target data for labeling prediction.
    - threshold: Threshold for correct rating predictions.
    - input_text_list: List of input text descriptions.

    Returns:
    - data_for_comparison: List of dictionaries containing sample predictions and actual values.
    - correct_num_list: List containing the number of correct predictions for each dimension.
    - all_correct: Total number of samples with all correct predictions.
    """
    device = next(model1.parameters()).device
    model1.eval()
    model2.eval()
    data_for_comparison = []
    correct_num_list = [0, 0, 0, 0, 0, 0]
    all_correct = 0
    label_all_correct = 0

    criterion = nn.MSELoss()

    # Move input and targets to the correct device
    test_input, test_target1, test_target2 = test_input.to(device), test_target1.to(device), test_target2.to(device)

    with torch.no_grad():
        test_output1 = model1(test_input)
        test_output2 = model2(test_input)

        for i in range(len(test_input)):
            test_input_sample = test_input[i].cpu().detach().numpy().tolist()
            test_target_sample1 = test_target1[i].cpu().detach().numpy().tolist()
            test_target_sample2 = test_target2[i].cpu().detach().numpy().tolist()
            test_output_sample1 = test_output1[i].cpu().detach().numpy().tolist()
            test_output_sample2 = test_output2[i].cpu().detach().numpy().tolist()
            test_loss = criterion(test_output1[i], test_target1[i])
            output1_loss = test_loss.item()

            normalized_label = normalize_list(test_output_sample2)
            test_combined_result = [x * y for x, y in zip(test_output_sample1, normalized_label)]

            bool_list = compare_lists(test_combined_result, test_target_sample1, threshold)
            label_bool_list = compare_lists(convert_to_label(normalized_label), test_target_sample2, 0.1)
            correct_num_list = [x + 1 if bool_val else x for x, bool_val in zip(correct_num_list, bool_list)]
            if all(bool_list):
                all_correct += 1
            if all(label_bool_list):
                label_all_correct += 1

            current_data = {
                'scenario_and_description': input_text_list[i],
                'initial_predicted_values_rating': test_output_sample1,
                'predicted_values_rating': test_combined_result,
                'actual_values_rating': test_target_sample1,
                'initial_predicted_values_labeling': test_output_sample2,
                'predicted_values_labeling': normalized_label,
                'actual_values_labeling': test_target_sample2,
            }
            data_for_comparison.append(current_data)

    return data_for_comparison, correct_num_list, all_correct, label_all_correct

def test_and_return_combined_result_separate(model1, model2, scenario_input, test_input, test_target1, test_target2, threshold, input_text_list):
    device = next(model1.parameters()).device
    model1.eval()
    model2.eval()
    data_for_comparison = []
    correct_num_list = [0, 0, 0, 0, 0, 0]
    all_correct = 0
    label_all_correct = 0

    criterion = nn.MSELoss()

    # Move input and targets to the correct device
    test_input = test_input.to(device)
    if test_target1 is not None and test_target2 is not None:
        test_target1, test_target2 = test_target1.to(device), test_target2.to(device)

    with torch.no_grad():
        test_output1 = model1(test_input)
        test_output2 = model2(test_input)
        if test_target1 is None and test_target2 is None:
            test_target1 = torch.zeros_like(test_output1).to(device)
            test_target2 = torch.zeros_like(test_output2).to(device)
        test_output3 = model1(scenario_input)
        test_output4 = model2(scenario_input)
        test_output_sample3 = test_output3[0].cpu().detach().numpy().tolist()
        test_output_sample4 = test_output4[0].cpu().detach().numpy().tolist()
        normalized_label = normalize_list(test_output_sample4)
        scenario_test_combined_result = [x * y for x, y in zip(test_output_sample3, normalized_label)]

        for i in range(len(test_input)):
            test_input_sample = test_input[i].cpu().detach().numpy().tolist()
            test_target_sample1 = test_target1[i].cpu().detach().numpy().tolist()
            test_target_sample2 = test_target2[i].cpu().detach().numpy().tolist()
            test_output_sample1 = test_output1[i].cpu().detach().numpy().tolist()
            test_output_sample2 = test_output2[i].cpu().detach().numpy().tolist()
            test_loss = criterion(test_output1[i], test_target1[i])
            output1_loss = test_loss.item()

            normalized_label = normalize_list(test_output_sample2)
            test_combined_result = [x * y for x, y in zip(test_output_sample1, normalized_label)]

            bool_list = compare_lists(test_combined_result, test_target_sample1, threshold)
            label_bool_list = compare_lists(convert_to_label(normalized_label), test_target_sample2, 0.1)
            correct_num_list = [x + 1 if bool_val else x for x, bool_val in zip(correct_num_list, bool_list)]
            if all(bool_list):
                all_correct += 1
            if all(label_bool_list):
                label_all_correct += 1

            current_data = {
                'scenario_and_description': input_text_list[i],
                'scenario_predictied_rating': scenario_test_combined_result,
                'initial_predicted_values_rating': test_output_sample1,
                'predicted_values_rating': test_combined_result,
                'actual_values_rating': test_target_sample1,
                'initial_predicted_values_labeling': test_output_sample2,
                'predicted_values_labeling': normalized_label,
                'actual_values_labeling': test_target_sample2,
            }
            data_for_comparison.append(current_data)

    return data_for_comparison, correct_num_list, all_correct, label_all_correct

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
    model_included_value = get_specified_order_name([6, 1, 5, 2, 4, 3])

    # labeling_model_name = f"{labeling_model_type}_{data_mode}_avg_128_{labeling_learning_rate}_{labeling_weight_decay}_{threshold}"
    # rating_model_name = f"{rating_model_type}_{data_mode}_avg_128_{rating_learning_rate}_{rating_weight_decay}"

    # Specifying the start and end values
    start_value = 1
    end_value = 6

    # Initialize included_value
    included_value = ""

    # Looping over different numbers of values
    for num_value in range(start_value, end_value + 1):  # Looping from the specified start value to the end value
        data_mode = "test"
        data_dir = f"./preprocessed_dataset/{encoder_model}/value_{num_value}"
        input_test, target_rate, target_label = load_data_tensors(data_dir, "test")
        if num_value == start_value:
            # Assigning values directly for the first iteration
            combined_input_test = input_test
            combined_target_rate = target_rate
            combined_target_label = target_label
            combined_input_text_list = get_input_text_list(num_value)
            included_value += f"{num_value}"  # Update included_value
        else:
            # Concatenating data from the second iteration onwards
            combined_input_test = torch.cat([combined_input_test, input_test], dim=0)
            combined_target_rate = torch.cat([combined_target_rate, target_rate], dim=0)
            combined_target_label = torch.cat([combined_target_label, target_label], dim=0)
            input_text_list = get_input_text_list(num_value)
            combined_input_text_list = combined_input_text_list + input_text_list
            included_value += f"_{num_value}"  # Update included_value
    
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

    # Creating test dataset and loader
    test_dataset = TensorDataset(combined_input_test, combined_target_rate, combined_target_label)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Threshold for correct ratings
    correct_rating_threshold = 0.2

    # Output type
    output_type = "Combine rating and labeling"

    # Testing and returning results
    if output_type == "NOT Combine rating and labeling":
        result, correct_num_list, all_correct, label_all_correct = test_and_return_result(rate_model, label_model, combined_input_test,
                                                                       combined_target_rate, combined_target_label,
                                                                       correct_rating_threshold, combined_input_text_list)
    else:
        result, correct_num_list, all_correct, label_all_correct = test_and_return_combined_result(rate_model, label_model,
                                                                                combined_input_test,
                                                                                combined_target_rate,
                                                                                combined_target_label,
                                                                                correct_rating_threshold,
                                                                                combined_input_text_list)

    # Printing accuracy metrics
    print_accuracy(included_value, correct_num_list, all_correct, label_all_correct, len(combined_input_text_list),
                   correct_rating_threshold, output_type)

    # Selecting and printing sample results
    selected_samples = random_select_n_samples(result, n=5)
    print_and_save_results(selected_samples)