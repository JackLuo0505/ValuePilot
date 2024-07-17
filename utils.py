"""
Utils functions for data loading, model evaluation, and result visualization.
Author: Anonymous Author
Date: ---

This script contains utility functions for various tasks such as data loading, model evaluation, and result visualization.

In this script:
- set_seed(seed): Set the seed for random number generators for reproducibility.
- load_and_split_data_tensors(data_dir, mode, split_ratio): Load and split data tensors into training and validation sets.
- load_data_tensors(data_dir, mode): Load data tensors from files.
- multi_label_accuracy(predictions, labels, threshold_high, threshold_low, n): Calculate the multi-label accuracy.
- single_label_accuracy(predictions, labels, threshold_high, threshold_low): Calculate the single-label accuracy for each dimension.
- compare_lists(list1, list2, threshold): Compare two lists element-wise within a given threshold.
- normalize_list(input_list): Normalize a list of values between 0 and 1.
- random_select_n_samples(data_for_comparison, n): Randomly select n samples from a dataset for comparison.
- convert_to_label(values, threshold): Convert continuous values to binary labels based on a threshold.
- load_model_and_optimizer(model, optimizer, model_path, optimizer_path): Load the model and optimizer from saved state dictionaries.
- get_specified_order_name(specified_order): Convert a list of specified orders into a string format.
- evaluate_accuracy(device, rate_model, data_loader, threshold=0.2): Evaluate the accuracy of a rate model on a given dataset.
- compute_loss(values, labels, targets, criterion): Compute the loss for the rate model.
- print_accuracy(included_value, correct_num_list, all_correct, label_all_correct, num_inputs, threshold, output_type = "Combine rating and labeling"): Print and display accuracy results.
- print_and_save_results(selected_samples, save_dir): Print and save inference results.
- save_pentagon_plot(predicted_values, actual_values, save_path, feature_labels, center, outer): Save a pentagon plot comparing predicted and actual values.
"""

import os
import torch
import random
import textwrap
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed):
    """
    Set the seed for random number generators for reproducibility.

    Args:
    - seed (int): Seed value.

    Returns:
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_and_split_data_tensors(data_dir, mode, split_ratio=0.8, chunks_num=''):
    """
    Load and split data tensors into training and validation sets.

    Args:
    - data_dir (str): Directory containing the data tensors.
    - mode (str): Mode of the dataset (train, test, validation).
    - split_ratio (float): Ratio of training data to total data.

    Returns:
    - X_train, y_train, z_train (torch.Tensor): Training data tensors.
    - X_validation, y_validation, z_validation (torch.Tensor): Validation data tensors.
    """
    X_path = os.path.join(data_dir, f'{mode}_data', f'X_AllData{chunks_num}.pt')
    y_path = os.path.join(data_dir, f'{mode}_data', f'y_AllData{chunks_num}.pt')
    z_path = os.path.join(data_dir, f'{mode}_data', f'z_AllData{chunks_num}.pt')

    # Load tensors
    X_AllData_tensor = torch.load(X_path)
    y_AllData_tensor = torch.load(y_path)
    z_AllData_tensor = torch.load(z_path)
    z_AllData_tensor = z_AllData_tensor.to(dtype=torch.float32)
    
    # Generate random permutation of indices
    indices = torch.randperm(len(X_AllData_tensor))

    # Shuffle tensors using shuffled indices
    X_AllData_tensor = X_AllData_tensor[indices]
    y_AllData_tensor = y_AllData_tensor[indices]
    z_AllData_tensor = z_AllData_tensor[indices]

    # Calculate split index
    split_index = int(len(indices) * split_ratio)

    # Split tensors
    X_train, X_validation = X_AllData_tensor[:split_index], X_AllData_tensor[split_index:]
    y_train, y_validation = y_AllData_tensor[:split_index], y_AllData_tensor[split_index:]
    z_train, z_validation = z_AllData_tensor[:split_index], z_AllData_tensor[split_index:]

    return X_train, y_train, z_train, X_validation, y_validation, z_validation

def load_data_tensors(data_dir, mode):
    """
    Load data tensors from files.

    Args:
    - data_dir (str): Directory containing the data tensors.
    - mode (str): Mode of the dataset (train, test, validation).

    Returns:
    - X_AllData_tensor (torch.Tensor): Input data tensor.
    - y_AllData_tensor (torch.Tensor): Output data tensor.
    - z_AllData_tensor (torch.Tensor): Label data tensor.
    """
    X_path = os.path.join(data_dir, f'{mode}_data', 'X_AllData.pt')
    y_path = os.path.join(data_dir, f'{mode}_data', 'y_AllData.pt')
    z_path = os.path.join(data_dir, f'{mode}_data', 'z_AllData.pt')

    # Load tensors
    X_AllData_tensor = torch.load(X_path)
    y_AllData_tensor = torch.load(y_path)
    z_AllData_tensor = torch.load(z_path)
    z_AllData_tensor = z_AllData_tensor.to(dtype=torch.float32)

    return X_AllData_tensor, y_AllData_tensor, z_AllData_tensor

def multi_label_accuracy(predictions, labels, threshold_high=0.8, threshold_low=0.2, n=6):
    """
    Calculate the multi-label accuracy.

    Args:
    - predictions (torch.Tensor): Predicted values.
    - labels (torch.Tensor): Actual values.
    - threshold_high (float): High threshold for prediction.
    - threshold_low (float): Low threshold for prediction.
    - n (int): Minimum number of correct labels for each sample.

    Returns:
    - accuracy (float): Multi-label accuracy.
    """
    preds = torch.zeros_like(predictions)
    preds[predictions > threshold_high] = 1
    preds[(predictions >= threshold_low) & (predictions <= threshold_high)] = -1  # 将不确定的预测标记为-1

    # Ensure preds and labels are binary (0 or 1)
    preds = preds.int()
    labels = labels.int()

    # Calculate the number of matching bits per sample
    matches_per_sample = (preds == labels).sum(dim=1)

    # Check if the number of matching bits for each sample reaches or exceeds n
    correct_samples = (matches_per_sample >= n).sum().item()

    # Calculate accuracy
    accuracy = correct_samples / preds.size(0)
    return accuracy

def single_label_accuracy(predictions, labels, threshold_high=0.8, threshold_low=0.2):
    """
    Calculate the single-label accuracy for each dimension.

    Args:
    - predictions (torch.Tensor): Predicted values.
    - labels (torch.Tensor): Actual values.
    - threshold_high (float): High threshold for prediction.
    - threshold_low (float): Low threshold for prediction.

    Returns:
    - accuracy_per_label (torch.Tensor): Single-label accuracy for each dimension.
    """
    preds = torch.zeros_like(predictions)
    preds[predictions > threshold_high] = 1
    preds[(predictions >= threshold_low) & (predictions <= threshold_high)] = -1  # 将不确定的预测标记为-1

    # Ensure preds and labels are binary (0 or 1)
    preds = preds.int()
    labels = labels.int()

    # Calculate correctness for each label
    correct_per_label = (preds == labels) & (preds != -1)

    # Calculate accuracy for each dimension
    accuracy_per_label = correct_per_label.sum(dim=0).float() / (labels != -1).sum(dim=0).float()

    return accuracy_per_label

def compare_lists(list1, list2, threshold=0.2):
    """
    Compare two lists element-wise within a given threshold.

    Args:
    - list1 (list): First list for comparison.
    - list2 (list): Second list for comparison.
    - threshold (float): Threshold for comparison.

    Returns:
    - comparison_result (list): List of comparison results.
    """
    return [abs(x - y) < threshold for x, y in zip(list1, list2)]

def normalize_list(input_list):
    """
    Normalize a list of values between 0 and 1.

    Args:
    - input_list (list): Input list of values.

    Returns:
    - normalized_list (list): Normalized list of values.
    """
    min_value = 0
    max_value = max(input_list)
    normalized_list = [(value - min_value) / (max_value - min_value) for value in input_list]
    return normalized_list

def random_select_n_samples(data_for_comparison, n):
    """
    Randomly select n samples from a dataset for comparison.

    Args:
    - data_for_comparison (list): Dataset for comparison.
    - n (int): Number of samples to select.

    Returns:
    - selected_samples (list): List of randomly selected samples.
    """
    return random.sample(data_for_comparison, n)

def convert_to_label(values, threshold=0.7):
    """
    Convert continuous values to binary labels based on a threshold.

    Args:
    - values (list): List of continuous values.
    - threshold (float): Threshold for conversion.

    Returns:
    - converted (list): List of binary labels.
    """
    # Ensure the threshold is between 0 and 1
    threshold = max(0, min(threshold, 1))

    # Convert each value in the list
    converted = []
    for value in values:
        if value > threshold:
            converted.append(1.0)
        elif value < 1 - threshold:
            converted.append(0.0)
        else:
            converted.append(0.5)

    return converted

def load_model_and_optimizer(model, optimizer, model_path, optimizer_path):
    """
    Load the model and optimizer from saved state dictionaries.

    Args:
    - model (torch.nn.Module): Model to load the state dictionary into.
    - optimizer (torch.optim.Optimizer): Optimizer to load the state dictionary into.
    - model_path (str): Path to the saved model state dictionary.
    - optimizer_path (str): Path to the saved optimizer state dictionary.

    Returns:
    - model (torch.nn.Module): Loaded model.
    - optimizer (torch.optim.Optimizer): Loaded optimizer.
    """
    # Load the model's state dictionary
    model.load_state_dict(torch.load(model_path))
    
    # Load the optimizer's state dictionary
    optimizer.load_state_dict(torch.load(optimizer_path))
    
    # Make sure to move the model to the correct device if needed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Put the model in evaluation mode (important if it has layers like dropout)
    model.eval()
    
    return model, optimizer

def get_specified_order_name(specified_order):
    """
    This function converts a list of specified orders into a string format.

    Args:
    - specified_order (list): A list containing integers representing specified orders.

    Returns:
    - specified_order_name (str): A string containing specified orders separated by underscores.
    """
    number_to_name = {
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6"
    }

    specified_order_name = "_".join(number_to_name[num] for num in specified_order)
    return specified_order_name

def evaluate_accuracy(device, rate_model, data_loader, threshold=0.2):
    """
    This function evaluates the accuracy of a rate model on a given dataset.

    Args:
    - device: The device (CPU or GPU) on which the model should be evaluated.
    - rate_model: The rate model to be evaluated.
    - data_loader: The DataLoader object containing the dataset.
    - threshold (float, optional): Threshold value for comparison. Defaults to 0.2.

    Returns:
    - accuracies (list): A list containing accuracies for each class.
    - all_correct_accuracy (float): Overall accuracy.
    """
    rate_model.eval()
    correct_num_list = [0, 0, 0, 0, 0, 0]
    all_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, target_rate, target_label in data_loader:
            X_batch, target_rate, target_label = X_batch.to(device), target_rate.to(device), target_label.to(device)
            output_label = target_label
            output_rate = rate_model(X_batch)

            # Process the output for each sample
            for i in range(X_batch.size(0)):
                output_label_sample = output_label[i].cpu().numpy().tolist()
                output_rate_sample = output_rate[i].cpu().numpy().tolist()
                target_rate_sample = target_rate[i].cpu().numpy().tolist()

                # Calculate combined output by element-wise multiplication
                combined_output_sample = [x * y for x, y in zip(output_rate_sample, output_label_sample)]
                bool_list = compare_lists(combined_output_sample, target_rate_sample, threshold)

                correct_num_list = [x + int(bool_val) for x, bool_val in zip(correct_num_list, bool_list)]
                if all(bool_list):
                    all_correct += 1

            total_samples += X_batch.size(0)

    accuracies = [correct_count / total_samples for correct_count in correct_num_list]
    all_correct_accuracy = all_correct / total_samples
    return accuracies, all_correct_accuracy

def compute_loss(values, labels, targets, criterion):
    """
    This function computes the loss for the rate model.

    Args:
    - values (tensor): Output values from the rate model.
    - labels (tensor): Output labels.
    - targets (tensor): Target values.
    - criterion: The loss function criterion.

    Returns:
    - loss: The computed loss value.
    """
    adjusted_values = values * (0.5 + 0.5*labels)
    loss = criterion(adjusted_values, targets)
    return loss

def print_accuracy(included_value, correct_num_list, all_correct, label_all_correct, num_inputs, threshold, output_type="Combine rating and labeling"):
    """
    Print and display accuracy results as a table suitable for Excel.

    Args:
    - included_value (str): Description of included value dimensions.
    - correct_num_list (list): List of correct counts for each dimension.
    - all_correct (int): Total count of fully correct predictions.
    - label_all_correct (int): Total count of fully correct label predictions.
    - num_inputs (int): Total number of input samples.
    - threshold (float): Threshold for correct rating.
    - output_type (str): Type of output to display.

    Returns:
    None
    """
    included_value = included_value.replace("_", ", ")
    accuracies = [correct_count / num_inputs for correct_count in correct_num_list]
    all_correct_accuracy = all_correct / num_inputs
    label_all_correct_accuracy = label_all_correct / num_inputs

    dimension_order = ['curiosity', 'energy', 'safety', 'happiness', 'intimacy', 'fairness']
    
    # Prepare the data for tabular display
    data = [
        ["Metric", "Value"],
        ["Included Data", included_value],
        ["Output Type", output_type],
        ["Threshold", threshold],
        ["Label Full Correct Accuracy", f"{label_all_correct_accuracy * 100:.2f}%"],
        ["Full Correct Accuracy", f"{all_correct_accuracy * 100:.2f}%"]
    ]
    
    for i, dim in enumerate(dimension_order):
        data.append([f"{dim.capitalize()} Accuracy", f"{accuracies[i] * 100:.2f}%"])
    
    data.extend([
        ["Label Full Correct Number", label_all_correct],
        ["Full Correct Number", all_correct],
        ["Number of Inputs", num_inputs]
    ])
    
    for i, dim in enumerate(dimension_order):
        data.append([f"{dim.capitalize()} Correct Number", correct_num_list[i]])
    
    # Format the data for Excel
    table = "\n".join([",".join(map(str, row)) for row in data])
    
    # Print the table
    print(table)

def print_and_save_results(selected_samples, save_dir="./inference_results"):
    """
    Print and save inference results.

    Args:
    - selected_samples (list): List of selected samples for inference.
    - save_dir (str): Directory to save results.

    Returns:
    None
    """
    # Create the directory to store the results
    os.makedirs(save_dir, exist_ok=True)

    for idx, data in enumerate(selected_samples):
        # Print the results
        text = data['scenario_and_description']
        max_line_length = 80  # Maximum length per line

        # Use the textwrap module for automatic line wrapping
        formatted_text = textwrap.fill(text, width=max_line_length)

        print(f"Scenario and Description:\n{formatted_text}")
        
        print("Rating:")
        initial_predicted_rating = [round(value, 3) for value in data['initial_predicted_values_rating']]
        predicted_rating = [round(value, 3) for value in data['predicted_values_rating']]
        actual_rating = [round(value, 3) for value in data['actual_values_rating']]
        print(f"  Initial Predicted Values: {initial_predicted_rating}")
        print(f"  Predicted Values: {predicted_rating}")
        print(f"  Actual Values:    {actual_rating}")
        
        print("Labeling:")
        initial_predicted_labeling = [round(value, 3) for value in data['initial_predicted_values_labeling']]
        predicted_labeling = convert_to_label(data['predicted_values_labeling'])
        actual_labeling = [round(value, 3) for value in data['actual_values_labeling']]
        print(f"  Initial Predicted Values: {initial_predicted_labeling}")
        print(f"  Predicted Values: {predicted_labeling}")
        print(f"  Actual Values:    {actual_labeling}")
        
        print("=" * 80)

        # Save the pentagon plots
        save_pentagon_plot(data['predicted_values_rating'], data['actual_values_rating'], os.path.join(save_dir, f"rating_result_{idx}.png"))
        save_pentagon_plot(data['predicted_values_labeling'], data['actual_values_labeling'], os.path.join(save_dir, f"labeling_result_{idx}.png"))

def save_pentagon_plot(predicted_values, actual_values, save_path, feature_labels = ['curiosity', 'energy', 'safety', 'happiness', 'intimacy', 'fairness'], center=-1, outer=1):
    """
    Save a pentagon plot comparing predicted and actual values.

    Args:
    - predicted_values (list): Predicted values.
    - actual_values (list): Actual values.
    - save_path (str): Path to save the plot.
    - feature_labels (list): Labels for each feature dimension.
    - center (float): Center value for normalization.
    - outer (float): Outer value for normalization.

    Returns:
    None
    """
    # Angle settings
    angles = np.linspace(0, 2 * np.pi, len(feature_labels), endpoint=False)

    # Calculate normalized values between center and outer
    normalized_predicted = [(value - center) / (outer - center) for value in predicted_values]
    normalized_actual = [(value - center) / (outer - center) for value in actual_values]

    # Plot the pentagon graph
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, normalized_predicted, color='red', alpha=0.25, label='Predicted Values')
    ax.fill(angles, normalized_actual, color='blue', alpha=0.25, label='Actual Values')

    # Set labels
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(feature_labels)

    # Display values for each point
    for i, angle in enumerate(angles):
        ax.text(angle, normalized_predicted[i], '{:.2f}'.format(predicted_values[i]), color='red', ha='left', va='bottom')
        ax.text(angle, normalized_actual[i], '{:.2f}'.format(actual_values[i]), color='blue', ha='left', va='top')

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.12))

    # Save the plot
    plt.savefig(save_path)
    plt.close()