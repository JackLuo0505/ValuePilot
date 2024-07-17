"""
Adaptive Complexity Curriculum Learning (ACCL) Training Script for Cascaded Judgment Rating Network (CJRN)
Author: Anonymous Author
Date: ---

This script trains a CJRN which combines the outputs of two parallel networks, LabelNet (LN) and RateNet (RN), to produce a final result. The real label values are loaded and used as controls for the rate outputs during training.

In this script:
- The CJRN architecture is defined and trained.
- Early stopping mechanism is employed to prevent overfitting.
- Training progress is logged using WandB.
"""

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

if __name__ == '__main__':
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--threshhold', type=float, default=0.8, help='threshold for multi_label_accuracy')
    parser.add_argument('-e', '--encoder_model', type=str, default='t5-base', help='Encoder model used',
                        choices=['t5-small', 't5-base', 'flan-t5-base', 'bert-base-uncased', 'roberta-base'])
    parser.add_argument('--use_wandb', type=int, default=1, help='Use wandb for logging')
    parser.add_argument('-c', '--num_chunks', type=int, default=4, help='Number of chunks to split the data into')
    parser.add_argument('--entity', type=str, default="value_agents", help='Wandb entity name')
    threshold = parser.parse_args().threshhold
    encoder_model = parser.parse_args().encoder_model
    use_wandb = parser.parse_args().use_wandb
    num_chunks = parser.parse_args().num_chunks
    entity = parser.parse_args().entity

    # Setting device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Model configuration
    rating_model_type = "ACCL_CJRN"
    data_mode = "gpt4"
    rating_learning_rate = 1e-5
    rating_weight_decay = 1e-4

    rating_model_name = f"ACCL_CJRN_{encoder_model}" 

    # Initializing Weights and Biases
    if use_wandb:
        wandb.init(project="value_driven", entity=entity, name=rating_model_name)

    # Initializing the RateNet model
    if encoder_model == 't5-base' or encoder_model == 'flan-t5-base' or encoder_model == 'bert-base-uncased' or encoder_model == 'roberta-base':
        rate_model = RateNet(768, 4, 128).to(device)
    elif encoder_model == 't5-small':
        rate_model = RateNet(512, 4, 128).to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rate_model.parameters(), lr=rating_learning_rate, weight_decay=rating_weight_decay)

    # Define the specified order
    specified_order = [6, 1, 5, 2, 4, 3]
    model_included_value = get_specified_order_name(specified_order)
    start_value = specified_order[0]
    
    # Training loop for different number of values
    for i in range(num_chunks):
        torch.cuda.empty_cache()
        # datasets
        data_dir = f"./preprocessed_dataset/{encoder_model}/value_{start_value}"

        # Loading and preparing test dataset
        input_test, target_test, target_test_label = load_data_tensors(data_dir, "test")
        input_train, target_train, target_train_label, input_validation, target_validation, target_validation_label = load_and_split_data_tensors(data_dir, "train", split_ratio=0.8, chunks_num=f"_{i+1}")

        initial_train_dataset = TensorDataset(input_train, target_train, target_train_label)
        initial_validation_dataset = TensorDataset(input_validation, target_validation, target_validation_label)
        initial_test_dataset = TensorDataset(input_test, target_test, target_test_label)

        combined_test_dataset = ConcatDataset([initial_test_dataset])
        combined_train_dataset = ConcatDataset([initial_train_dataset])
        combined_validation_dataset = ConcatDataset([initial_validation_dataset])
        included_value = "value"

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

            input_test, target_test, target_test_label = load_data_tensors(data_dir, "test")
            input_train, target_train, target_train_label, input_validation, target_validation, target_validation_label = load_and_split_data_tensors(data_dir, "train", split_ratio=0.8, chunks_num=f"_{i+1}")

            train_dataset = TensorDataset(input_train, target_train, target_train_label)
            validation_dataset = TensorDataset(input_validation, target_validation, target_validation_label)
            test_dataset = TensorDataset(input_test, target_test, target_test_label)

            # Combining test datasets
            if num_value != start_value:
                combined_train_dataset = ConcatDataset([combined_train_dataset, train_dataset])
                combined_validation_dataset = ConcatDataset([combined_validation_dataset, validation_dataset])
                combined_test_dataset = ConcatDataset([combined_test_dataset, test_dataset])

            train_loader = DataLoader(combined_train_dataset, batch_size=512, shuffle=True)
            validation_loader = DataLoader(combined_validation_dataset, batch_size=256, shuffle=True)

            early_stopping_patience = 10
            best_val_loss = float('inf')
            no_improvement_count = 0

            # Training loop
            n_epoch = 2000
            for epoch in range(n_epoch):
                total_loss = 0.0
                num_batches = 0

                # Training loop
                rate_model.train()
                for X_batch, y_batch, z_batch in train_loader:
                    X_batch, y_batch, z_batch = X_batch.to(device), y_batch.to(device), z_batch.to(device)

                    optimizer.zero_grad()
                    predicted_values = rate_model(X_batch)
                    predicted_labels = z_batch
                    loss = compute_loss(predicted_values, predicted_labels, y_batch, criterion)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                average_loss = total_loss / num_batches
                train_accs, train_all_correct_acc = evaluate_accuracy(device, rate_model, train_loader)

                # Validation loop
                rate_model.eval()
                validation_loss = 0.0
                num_validation_batches = 0

                for val_input, val_target, val_target_label in validation_loader:
                    val_input, val_target, val_target_label = val_input.to(device), val_target.to(device), val_target_label.to(device)

                    val_output_value = rate_model(val_input)
                    val_loss = compute_loss(val_output_value, val_target_label, val_target, criterion)

                    validation_loss += val_loss.item()
                    num_validation_batches += 1

                average_validation_loss = validation_loss / num_validation_batches

                validation_accs, validation_all_correct_acc = evaluate_accuracy(device, rate_model, validation_loader)

                # Logging and early stopping
                print(f"Epoch {epoch+1}/{n_epoch}, Training Loss: {average_loss}, Validation Loss: {average_validation_loss}, Validation Accuracy: {validation_all_correct_acc}")
                wandb.log({"avg_training_loss": average_loss, "avg_validation_loss": average_validation_loss, "avg_validation_acc": validation_all_correct_acc})

                # Check for early stopping
                if average_validation_loss < best_val_loss:
                    best_val_loss = average_validation_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1} as there is no improvement in validation loss.")
                    # save
                    torch.save(rate_model.state_dict(), os.path.join(save_path, f'rate_model.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(save_path, f'rate_optimizer.pth'))
                    break

    # Final evaluation on the test set
    rate_model.eval()
    test_loss = 0.0
    num_test_batches  = 0
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
    wandb.log({"avg_test_loss": average_test_loss, "test_accuracy": all_correct_acc})

    wandb.finish()