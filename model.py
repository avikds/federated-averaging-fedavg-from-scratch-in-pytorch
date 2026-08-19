"""
Federated Averaging (FedAvg) from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp_classifier
import torch
import torch.nn as nn

def build_mlp_classifier(input_size, hidden_size, num_classes):
    # Single hidden-layer MLP with ReLU activation.
    # The final layer returns raw logits (no softmax).
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    )

    return model

# Step 2 - build_synthetic_dataset
def build_synthetic_dataset(num_samples, input_size, num_classes, seed):
    # Create a dedicated generator so all randomness is controlled by seed.
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Generate float features and integer class labels.
    features = torch.randn(
        num_samples,
        input_size,
        generator=generator
    )

    labels = torch.randint(
        low=0,
        high=num_classes,
        size=(num_samples,),
        generator=generator,
        dtype=torch.long
    )

    return features, labels

# Step 3 - train_test_split_dataset (not yet solved)
# TODO: implement

# Step 4 - partition_data_iid (not yet solved)
# TODO: implement

# Step 5 - partition_data_non_iid (not yet solved)
# TODO: implement

# Step 6 - count_client_samples (not yet solved)
# TODO: implement

# Step 7 - iterate_client_batches (not yet solved)
# TODO: implement

# Step 8 - compute_batch_loss (not yet solved)
# TODO: implement

# Step 9 - local_sgd_step (not yet solved)
# TODO: implement

# Step 10 - train_client_local (not yet solved)
# TODO: implement

# Step 11 - clone_model_state (not yet solved)
# TODO: implement

# Step 12 - load_model_state (not yet solved)
# TODO: implement

# Step 13 - initialize_global_state (not yet solved)
# TODO: implement

# Step 14 - add_state_dicts (not yet solved)
# TODO: implement

# Step 15 - scale_state_dict (not yet solved)
# TODO: implement

# Step 16 - aggregate_weighted_average (not yet solved)
# TODO: implement

# Step 17 - select_round_clients (not yet solved)
# TODO: implement

# Step 18 - run_communication_round (not yet solved)
# TODO: implement

# Step 19 - evaluate_accuracy (not yet solved)
# TODO: implement

# Step 20 - run_fedavg (not yet solved)
# TODO: implement

# Step 21 - train_centralized_baseline (not yet solved)
# TODO: implement

# Step 22 - run_fedavg_iid (not yet solved)
# TODO: implement

# Step 23 - run_fedavg_non_iid (not yet solved)
# TODO: implement

# Step 24 - compute_non_iid_gap (not yet solved)
# TODO: implement

# Step 25 - rounds_to_target_vs_local_epochs (not yet solved)
# TODO: implement

# Step 26 - accuracy_vs_client_fraction (not yet solved)
# TODO: implement

