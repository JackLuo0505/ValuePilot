"""
Training Script for Labeling Model
Author: Anonymous Author
Date: ---

This script trains a labeling model using PyTorch and logs the training process and results using Weights and Biases (wandb).

In this script:
- Data loading and preprocessing are handled using utility functions.
- The labeling model architecture is defined.
- Training loop is implemented with early stopping and model checkpointing.
- Training and validation metrics are logged using wandb.
- Final evaluation is performed on the test set, and results are logged.
"""

# Importing libraries
import torch
import torch.nn as nn
from utils import set_seed

# Define the Labeling Model architecture
class LabelNet(nn.Module):
    def __init__(self, hidden_size=768, num_heads=4, mlp_hidden_size=128):
        super(LabelNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_size = mlp_hidden_size

        # self attention
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_heads)

        # hidden size Fully connected layers
        self.hidden_fc1 = nn.Linear(self.hidden_size, self.mlp_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.hidden_fc2 = nn.Linear(self.mlp_hidden_size, 6)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)

        # self attention
        attn_output, _ = self.attention(x, x, x)  # (seq_len, batch_size, hidden_size)

        # average pooling
        x = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
        x = torch.mean(x, dim=1)

        # MLP on hidden layers
        x = self.hidden_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden_fc2(x)

        return torch.sigmoid(x)
    
class RateNet(nn.Module):
    def __init__(self, hidden_size=768, num_heads=4, mlp_hidden_size=128):
        super(RateNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_size = mlp_hidden_size

        # Self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_heads)

        # Fully connected layers
        self.hidden_fc1 = nn.Linear(self.hidden_size, self.mlp_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.hidden_fc2 = nn.Linear(self.mlp_hidden_size, 6)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)

        # Self-attention
        attn_output, _ = self.attention(x, x, x)  # (seq_len, batch_size, hidden_size)

        # Average pooling
        x = attn_output.permute(1, 0, 2)
        x = torch.mean(x, dim=1)

        # MLP layers
        x = self.hidden_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden_fc2(x)

        return torch.tanh(x)