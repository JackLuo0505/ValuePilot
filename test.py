# Importing libraries
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch
import torch.nn as nn
import wandb
import argparse
import os
from networks import RateNet
from utils import load_and_split_data_tensors, load_data_tensors, set_seed, evaluate_accuracy, compute_loss, get_specified_order_name

# Setting seed for reproducibility
set_seed(42)

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--threshhold', type=float, default=0.8, help='threshold for multi_label_accuracy')
parser.add_argument('--accuracy_mode', type=str, choices=['multi', 'single'], help='accuracy mode for evaluation')
parser.add_argument('-e', '--encoder_model', type=str, default='t5-base', help='Encoder model used',
                    choices=['t5-small', 't5-base', 'flan-t5-base', 'chatglm3-6b', 'bert-base-uncased', 'roberta-base', 'glm-4-9b', 'glm-4-9b-chat'])
parser.add_argument('--use_wandb', type=int, default=1, help='Use wandb for logging')
parser.add_argument('-c', '--num_chunks', type=int, default=4, help='Number of chunks to split the data into')
parser.add_argument('--entity', type=str, default="value_agents", help='Wandb entity name')
threshold = parser.parse_args().threshhold
accuracy_mode = parser.parse_args().accuracy_mode
encoder_model = parser.parse_args().encoder_model
use_wandb = parser.parse_args().use_wandb
num_chunks = parser.parse_args().num_chunks
entity = parser.parse_args().entity
# Final evaluation on the test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rate_model = RateNet(hidden_size=768, num_heads=4, mlp_hidden_size=128).to(device)
# load
rate_model.load_state_dict(torch.load('model_save/t5-base/value_6_1_5_2_4_3/rate_model_ACCL_CJRN_t5-base.pth'))
rate_model.eval()
test_loss = 0.0
num_test_batches  = 0

specified_order = [6, 1, 5, 2, 4, 3]
model_included_value = get_specified_order_name(specified_order)
start_value = specified_order[0]
criterion = nn.MSELoss()

for i in range(num_chunks):
    torch.cuda.empty_cache()
    # datasets
    data_dir = f"./preprocessed_dataset/{encoder_model}/value_{start_value}"

    # Loading and preparing test dataset
    input_test, target_test, target_test_label = load_data_tensors(data_dir, "test")

    initial_test_dataset = TensorDataset(input_test, target_test, target_test_label)

    combined_test_dataset = ConcatDataset([initial_test_dataset])

    included_value = "value"

    for num_value in specified_order:
        print(f"Round {i+1}: Learning data with {num_value} values included")
        included_value = f"{included_value}_{num_value}"

        # Loading and preparing datasets
        data_dir = f"./preprocessed_dataset/{encoder_model}/value_{num_value}"
        value_dimensions = ['curiosity', 'energy', 'safety', 'happiness', 'intimacy', 'fairness']

        input_test, target_test, target_test_label = load_data_tensors(data_dir, "test")

        test_dataset = TensorDataset(input_test, target_test, target_test_label)

        # Combining test datasets
        if num_value != start_value:
            combined_test_dataset = ConcatDataset([combined_test_dataset, test_dataset])

test_loader = DataLoader(combined_test_dataset, batch_size=256, shuffle=True)

for test_input, test_target, test_target_label in test_loader:
    test_input, test_target, test_target_label = test_input.to(device), test_target.to(device), test_target_label.to(device)

    test_output = rate_model(test_input)
    test_loss_batch = compute_loss(test_output, test_target_label, test_target, criterion)
    test_loss += test_loss_batch.item()
    num_test_batches += 1

average_test_loss = test_loss / num_test_batches
accs, all_correct_acc = evaluate_accuracy(device, rate_model, test_loader)
print(f"Average Test Loss: {average_test_loss}", f"Test Accuracy: {all_correct_acc}, {accs}")