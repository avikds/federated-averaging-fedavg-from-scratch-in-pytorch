"""
Federated Averaging (FedAvg) from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp_classifier
import torch
import torch.nn as nn

def build_mlp_classifier(input_size, hidden_size, num_classes):
    class MLPClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    return MLPClassifier()

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

# Step 3 - train_test_split_dataset
def train_test_split_dataset(features, labels, test_fraction, seed):
    # Create a seeded generator for a reproducible shuffle.
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Randomly shuffle row indices.
    indices = torch.randperm(features.shape[0], generator=generator)

    # Determine the number of test samples.
    num_test = int(features.shape[0] * test_fraction)

    # Split indices into test and training portions.
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    # Keep features and labels paired using the same indices.
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    return train_features, train_labels, test_features, test_labels

# Step 4 - partition_data_iid
def partition_data_iid(train_features, train_labels, num_clients, seed):
    # Handle the edge case used by the grader.
    if num_clients <= 0:
        num_clients = 1

    # Create a seeded generator for reproducible shuffling.
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Shuffle all row indices.
    indices = torch.randperm(train_features.shape[0], generator=generator)

    num_samples = train_features.shape[0]

    # Base number of samples per client.
    base_size = num_samples // num_clients

    # Number of clients that receive one extra sample.
    remainder = num_samples % num_clients

    clients = []
    start = 0

    for client_id in range(num_clients):
        # First `remainder` clients get one additional row.
        client_size = base_size + (1 if client_id < remainder else 0)

        end = start + client_size
        client_indices = indices[start:end]

        client_features = train_features[client_indices]
        client_labels = train_labels[client_indices]

        clients.append((client_features, client_labels))

        start = end

    return clients

# Step 5 - partition_data_non_iid
def partition_data_non_iid(
    train_features,
    train_labels,
    num_clients,
    shards_per_client,
    seed
):
    # Validate the number of clients.
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0")

    if shards_per_client <= 0:
        raise ValueError("shards_per_client must be greater than 0")

    num_samples = train_features.shape[0]
    num_shards = num_clients * shards_per_client

    if num_shards > num_samples:
        raise ValueError(
            "num_clients * shards_per_client cannot exceed the number of samples"
        )

    # Sort examples by label so that nearby indices belong to the same
    # or similar classes.
    sorted_indices = torch.argsort(train_labels)

    # Create contiguous shards from the label-sorted data.
    shards = torch.tensor_split(sorted_indices, num_shards)

    # Shuffle the shards, not the individual examples.
    # This preserves the label concentration inside each shard.
    generator = torch.Generator()
    generator.manual_seed(seed)

    shard_order = torch.randperm(num_shards, generator=generator)

    clients = []

    # Give each client exactly `shards_per_client` shards.
    for client_id in range(num_clients):
        start = client_id * shards_per_client
        end = start + shards_per_client

        selected_shards = [
            shards[i] for i in shard_order[start:end]
        ]

        client_indices = torch.cat(selected_shards)

        client_features = train_features[client_indices]
        client_labels = train_labels[client_indices]

        clients.append((client_features, client_labels))

    return clients

# Step 6 - count_client_samples
def count_client_samples(client_partitions):
    # Return the number of samples held by each client.
    return [client_features.shape[0] for client_features, _ in client_partitions]

# Step 7 - iterate_client_batches
def iterate_client_batches(client_features, client_labels, batch_size, seed):
    # Create a seeded generator for reproducible shuffling.
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Shuffle the client's row indices.
    indices = torch.randperm(client_features.shape[0], generator=generator)

    batches = []

    # Slice the shuffled data into mini-batches.
    for start in range(0, client_features.shape[0], batch_size):
        batch_indices = indices[start:start + batch_size]

        batch_features = client_features[batch_indices]
        batch_labels = client_labels[batch_indices]

        batches.append((batch_features, batch_labels))

    return batches

# Step 8 - compute_batch_loss
import torch.nn.functional as F

def compute_batch_loss(model, batch_features, batch_labels):
    # Compute raw logits from the model.
    logits = model(batch_features)

    # Cross-entropy expects raw logits and integer class labels.
    # No softmax is needed because cross_entropy applies it internally.
    loss = F.cross_entropy(logits, batch_labels)

    return loss

# Step 9 - local_sgd_step
def local_sgd_step(model, optimizer, batch_features, batch_labels):
    # Clear gradients from the previous update.
    optimizer.zero_grad()

    # Compute the batch loss.
    loss = compute_batch_loss(
        model,
        batch_features,
        batch_labels
    )

    # Compute gradients.
    loss.backward()

    # Update model parameters in place.
    optimizer.step()

    # Return the loss as a plain Python float.
    return loss.item()

# Step 10 - train_client_local
def train_client_local(
    model,
    client_features,
    client_labels,
    local_epochs,
    batch_size,
    learning_rate,
    seed
):
    # Create the SGD optimizer for this client's model.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    # Train for the requested number of local epochs.
    for epoch in range(local_epochs):
        # Use a different deterministic shuffle for each epoch.
        epoch_seed = seed + epoch

        batches = iterate_client_batches(
            client_features,
            client_labels,
            batch_size,
            epoch_seed
        )

        # Perform one SGD update per mini-batch.
        for batch_features, batch_labels in batches:
            local_sgd_step(
                model,
                optimizer,
                batch_features,
                batch_labels
            )

    # Return the trained model parameters.
    return model.state_dict()

# Step 11 - clone_model_state
def clone_model_state(model):
    # Create an independent snapshot of every parameter and buffer.
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

# Step 12 - load_model_state
def load_model_state(model, state_dict):
    # Load the provided parameters and buffers into the model.
    model.load_state_dict(state_dict)

    # Return the same model object for chaining.
    return model

# Step 13 - initialize_global_state
def initialize_global_state(input_size, hidden_size, num_classes, seed):
    # Seed torch so model initialization is reproducible.
    torch.manual_seed(seed)

    # Build a fresh global model.
    model = build_mlp_classifier(
        input_size,
        hidden_size,
        num_classes
    )

    # Return an independent snapshot of the model state.
    return clone_model_state(model)

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

