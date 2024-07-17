"""
Adaptive Complexity Curriculum Learning (ACCL) Training Script for Labeling Model
Author: Anonymous Author
Date: ---

This script implements Adaptive Complexity Curriculum Learning (ACCL), a strategy for progressively training models using data of increasing difficulty. It gradually introduces more challenging data to the model during training.

In this script:
- The model is trained using a curriculum learning approach, starting with simpler data and progressively including more difficult samples, such as starting with data containing only one value and gradually mixing in data containing one and two values, and so forth.
- Early stopping is employed to prevent overfitting during training.
- The trained model is evaluated on a separate test set, and performance metrics are logged using Weights & Biases (wandb).
"""

# Importing libraries
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch
import torch.nn as nn
import wandb
import numpy as np
import argparse
import os
from networks import LabelNet
from utils import set_seed, load_and_split_data_tensors, load_data_tensors, multi_label_accuracy, single_label_accuracy

# Setting seed for reproducibility
set_seed(42)

# Main script
if __name__ == '__main__':
    
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--threshhold', type=float, default=0.8, help='threshold for multi_label_accuracy')
    parser.add_argument('--accuracy_mode', type=str, default='multi', choices=['multi', 'single'], help='accuracy mode for evaluation')
    parser.add_argument('-e', '--encoder_model', type=str, default='t5-base', help='Encoder model used',
                        choices=['t5-small', 't5-base', 'flan-t5-base', 'bert-base-uncased', 'roberta-base'])
    parser.add_argument('--use_wandb', type=int, default=1, help='Use wandb for logging')
    parser.add_argument('-c', '--num_chunks', type=int, default=4, help='Number of chunks to split the data into')
    parser.add_argument('--entity', type=str, default="value_agents", help='Wandb entity name')
    threshold = parser.parse_args().threshhold
    accuracy_mode = parser.parse_args().accuracy_mode
    encoder_model = parser.parse_args().encoder_model
    use_wandb = parser.parse_args().use_wandb
    num_chunks = parser.parse_args().num_chunks
    entity = parser.parse_args().entity

    # Setting device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Model configuration
    learning_rate = 1e-3
    weight_decay = 1e-5

    # Initializing Weights and Biases
    if use_wandb:
        wandb.init(project="value_driven_label", entity=entity, name=f'ACCL_{encoder_model}_{accuracy_mode}')

    # Initializing the LabelNet model
    if encoder_model == 't5-base' or encoder_model == 'flan-t5-base' or encoder_model == 'bert-base-uncased' or encoder_model == 'roberta-base':
        label_model = LabelNet(768, 4, 128).to(device)
    elif encoder_model == 't5-small':
        label_model = LabelNet(512, 4, 128).to(device)
    # Defining loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(label_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define the specified order
    specified_order = [6,1,5,2,4,3]
    start_value = specified_order[0]

    # Loading and preparing test dataset
    data_dir = f"./preprocessed_dataset/{encoder_model}/value_{start_value}"
    
    for i in range(num_chunks):
        input_test, _, target_test = load_data_tensors(data_dir, "test")
        input_train, _, target_train, input_validation, _, target_validation = load_and_split_data_tensors(data_dir, "train", split_ratio=0.8, chunks_num=f"_{i+1}" if num_chunks > 1 else "")

        initial_test_dataset = TensorDataset(input_test, target_test)
        initial_train_dataset = TensorDataset(input_train, target_train)
        initial_validation_dataset = TensorDataset(input_validation, target_validation)

        combined_test_dataset = ConcatDataset([initial_test_dataset])
        combined_train_dataset = ConcatDataset([initial_train_dataset])
        combined_validation_dataset = ConcatDataset([initial_validation_dataset])

        included_value = "value"

        # Training loop for different number of values
        for num_value in specified_order:
            print(f"Round {i+1}: Learning data with {num_value} values included")
            included_value = f"{included_value}_{num_value}"

            # Creating save path for model
            save_path = f'./model_save/{encoder_model}/{included_value}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Loading and preparing datasets
            data_dir = f"./preprocessed_dataset/{encoder_model}/value_{num_value}"
            value_dimensions = ['curiosity', 'energy', 'safety', 'happiness', 'intimacy', 'fairness']

            # Loading and preparing datasets
            input_test, _, target_test = load_data_tensors(data_dir, "test")
            input_train, _, target_train, input_validation, _, target_validation = load_and_split_data_tensors(data_dir, "train", split_ratio=0.8, chunks_num=f"_{i+1}" if num_chunks > 1 else "")

            train_dataset = TensorDataset(input_train, target_train)
            validation_dataset = TensorDataset(input_validation, target_validation)
            test_dataset = TensorDataset(input_test, target_test)
            
            # Combining test datasets
            if num_value != start_value:
                combined_train_dataset = ConcatDataset([combined_train_dataset, train_dataset])
                combined_validation_dataset = ConcatDataset([combined_validation_dataset, validation_dataset])
                combined_test_dataset = ConcatDataset([combined_test_dataset, test_dataset])

            train_loader = DataLoader(combined_train_dataset, batch_size=512, shuffle=True)
            validation_loader = DataLoader(combined_validation_dataset, batch_size=256, shuffle=True)

            # Early stopping parameters
            early_stopping_patience = 10
            best_val_loss = float('inf')
            no_improvement_count = 0

            # Training loop
            n_epoch = 4000
            for epoch in range(n_epoch):
                total_loss = 0.0
                num_batches = 0
                accs = []

                # Training loop
                label_model.train()
                for X_batch, z_batch in train_loader:
                    X_batch, z_batch = X_batch.to(device), z_batch.to(device)

                    optimizer.zero_grad()
                    predicted_values = label_model(X_batch)
                    loss = criterion(predicted_values, z_batch)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    # Calculating accuracy
                    if accuracy_mode == "single":
                        train_accuracy = single_label_accuracy(predicted_values.detach(), z_batch)
                    elif accuracy_mode == "multi":
                        train_accuracy = multi_label_accuracy(predicted_values.detach(), z_batch)

                    accs.append(train_accuracy)

                average_loss = total_loss / num_batches
                mean_train_acc = np.mean(accs)

                # Validation loop
                label_model.eval()
                validation_loss = 0.0
                num_validation_batches = 0
                accs = []

                for val_input, val_target in validation_loader:
                    val_input, val_target = val_input.to(device), val_target.to(device)

                    val_output = label_model(val_input)
                    val_loss = criterion(val_output, val_target)

                    validation_loss += val_loss.item()
                    num_validation_batches += 1

                    # Calculating accuracy
                    if accuracy_mode == "single":
                        val_accuracy = single_label_accuracy(val_output.detach(), val_target)
                    elif accuracy_mode == "multi":
                        val_accuracy = multi_label_accuracy(val_output.detach(), val_target)

                    accs.append(val_accuracy)

                average_validation_loss = validation_loss / num_validation_batches
                mean_validation_acc = np.mean(accs)

                # Logging and early stopping
                print(f"Epoch {epoch+1}/{n_epoch}, Training Loss: {average_loss}, Training Accuracy: {mean_train_acc}, Validation Loss: {average_validation_loss}, Validation Accuracy: {mean_validation_acc}")
                if use_wandb:
                    wandb.log({"avg_training_loss": average_loss, "avg_validation_loss": average_validation_loss, "train_accuracy": mean_train_acc, "val_accuracy": mean_validation_acc})

                # Check for early stopping
                if average_validation_loss < best_val_loss:
                    best_val_loss = average_validation_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1} as there is no improvement in validation loss.")
                    # Save model
                    torch.save(label_model.state_dict(), os.path.join(save_path, f'label_model.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(save_path, f'label_optimizer.pth'))
                    break  # Early stopping

    # Final evaluation on the test set
    label_model.eval()
    test_loss = 0.0
    num_test_batches = 0
    accs = []
    test_loader = DataLoader(combined_test_dataset, batch_size=256, shuffle=True)

    # Test loop
    for test_input, test_target in test_loader:
        test_input, test_target = test_input.to(device), test_target.to(device)

        test_output = label_model(test_input)
        test_loss_batch = criterion(test_output, test_target)
        test_loss += test_loss_batch.item()
        num_test_batches += 1

        # Calculating accuracy
        if accuracy_mode == "single":
            test_accuracy = single_label_accuracy(test_output.detach(), test_target)
        elif accuracy_mode == "multi":
            test_accuracy = multi_label_accuracy(test_output.detach(), test_target)
        accs.append(test_accuracy)

    average_test_loss = test_loss / num_test_batches
    mean_acc = np.mean(accs)
    print(f"Average Test Loss: {average_test_loss}, Test Accuracy: {mean_acc}")
    if use_wandb:
        wandb.log({"avg_test_loss": average_test_loss, "test_accuracy": mean_acc})
        wandb.finish()
