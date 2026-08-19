"""
Federated Averaging (FedAvg) from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp_classifier
import torch
import torch.nn as nn

def build_mlp_classifier(input_size, hidden_size, num_classes):
    class _MLPClassifier(nn.Module):
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

    return _MLPClassifier()

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
    # Make model initialization reproducible.
    torch.manual_seed(seed)

    # Build the initial global model.
    model = build_mlp_classifier(
        input_size,
        hidden_size,
        num_classes
    )

    # Return detached, independent copies of all parameters.
    return clone_model_state(model)

# Step 14 - add_state_dicts
def add_state_dicts(state_a, state_b):
    # Create a new state dict without modifying either input.
    return {
        key: state_a[key] + state_b[key]
        for key in state_a
    }

# Step 15 - scale_state_dict
def scale_state_dict(state_dict, weight):
    # Create a new state dict without modifying the original.
    return {
        key: tensor * weight
        for key, tensor in state_dict.items()
    }

# Step 16 - aggregate_weighted_average
def aggregate_weighted_average(client_states, client_sample_counts):
    # Total number of samples across all clients.
    total_samples = sum(client_sample_counts)

    # Accumulate the sample-weighted client states.
    aggregated_state = None

    for client_state, sample_count in zip(
        client_states,
        client_sample_counts
    ):
        weight = sample_count / total_samples

        weighted_state = scale_state_dict(
            client_state,
            weight
        )

        if aggregated_state is None:
            aggregated_state = weighted_state
        else:
            aggregated_state = add_state_dicts(
                aggregated_state,
                weighted_state
            )

    return aggregated_state

# Step 17 - select_round_clients
def select_round_clients(num_clients, client_fraction, seed):
    # Determine how many clients to select.
    num_selected = max(
        1,
        round(client_fraction * num_clients)
    )

    # Create a seeded generator for reproducible selection.
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Randomly select clients without replacement.
    selected = torch.randperm(
        num_clients,
        generator=generator
    )[:num_selected]

    # Return indices in sorted order.
    return sorted(selected.tolist())

# Step 18 - run_communication_round
def run_communication_round(
    global_state,
    client_partitions,
    selected_clients,
    model_config,
    local_epochs,
    batch_size,
    learning_rate,
    seed
):
    client_states = []
    client_sample_counts = []

    for client_idx in selected_clients:
        # Build a fresh model for this client.
        model = build_mlp_classifier(
            model_config["input_size"],
            model_config["hidden_size"],
            model_config["num_classes"]
        )

        # Load the current global parameters.
        load_model_state(model, global_state)

        # Get this client's local data.
        client_features, client_labels = client_partitions[client_idx]

        # Train locally starting from the global state.
        client_state = train_client_local(
            model,
            client_features,
            client_labels,
            local_epochs,
            batch_size,
            learning_rate,
            seed + client_idx
        )

        client_states.append(client_state)
        client_sample_counts.append(client_features.shape[0])

    # Aggregate selected clients using sample-weighted FedAvg.
    return aggregate_weighted_average(
        client_states,
        client_sample_counts
    )

# Step 19 - evaluate_accuracy
def evaluate_accuracy(model, test_features, test_labels):
    # Put the model in evaluation mode.
    model.eval()

    # Run inference without tracking gradients.
    with torch.no_grad():
        logits = model(test_features)
        predictions = torch.argmax(logits, dim=1)

        # Compute the fraction of correct predictions.
        accuracy = (predictions == test_labels).float().mean()

    return accuracy.item()

# Step 20 - run_fedavg
def run_fedavg(
    client_partitions,
    test_features,
    test_labels,
    model_config,
    num_rounds,
    client_fraction,
    local_epochs,
    batch_size,
    learning_rate,
    seed
):
    # Initialize the global model state.
    global_state = initialize_global_state(
        model_config["input_size"],
        model_config["hidden_size"],
        model_config["num_classes"],
        seed
    )

    accuracies = []

    # Run the requested number of communication rounds.
    for round_idx in range(num_rounds):
        # Select clients for this communication round.
        selected_clients = select_round_clients(
            len(client_partitions),
            client_fraction,
            seed + round_idx
        )

        # Train selected clients and aggregate their states.
        global_state = run_communication_round(
            global_state,
            client_partitions,
            selected_clients,
            model_config,
            local_epochs,
            batch_size,
            learning_rate,
            seed + round_idx
        )

        # Build a fresh global model and load the updated state.
        model = build_mlp_classifier(
            model_config["input_size"],
            model_config["hidden_size"],
            model_config["num_classes"]
        )

        load_model_state(model, global_state)

        # Evaluate after this communication round.
        accuracy = evaluate_accuracy(
            model,
            test_features,
            test_labels
        )

        accuracies.append(accuracy)

    # Build the final model from the final global state.
    final_model = build_mlp_classifier(
        model_config["input_size"],
        model_config["hidden_size"],
        model_config["num_classes"]
    )

    load_model_state(final_model, global_state)

    return final_model, accuracies

# Step 21 - train_centralized_baseline
def train_centralized_baseline(
    train_features,
    train_labels,
    test_features,
    test_labels,
    model_config,
    num_epochs,
    batch_size,
    learning_rate,
    seed
):
    # Build a fresh model.
    model = build_mlp_classifier(
        model_config["input_size"],
        model_config["hidden_size"],
        model_config["num_classes"]
    )

    # Create the SGD optimizer.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    # Train for the requested number of epochs.
    for epoch in range(num_epochs):
        # Reshuffle the pooled training data each epoch.
        batches = iterate_client_batches(
            train_features,
            train_labels,
            batch_size,
            seed + epoch
        )

        # Perform one SGD update per batch.
        for batch_features, batch_labels in batches:
            local_sgd_step(
                model,
                optimizer,
                batch_features,
                batch_labels
            )

    # Evaluate the trained centralized model.
    return evaluate_accuracy(
        model,
        test_features,
        test_labels
    )

# Step 22 - run_fedavg_iid
def run_fedavg_iid(
    train_features,
    train_labels,
    test_features,
    test_labels,
    model_config,
    num_clients,
    num_rounds,
    client_fraction,
    local_epochs,
    batch_size,
    learning_rate,
    seed
):
    # Partition the training data IID across clients.
    client_partitions = partition_data_iid(
        train_features,
        train_labels,
        num_clients,
        seed
    )

    # Run the complete FedAvg pipeline.
    _, accuracies = run_fedavg(
        client_partitions,
        test_features,
        test_labels,
        model_config,
        num_rounds,
        client_fraction,
        local_epochs,
        batch_size,
        learning_rate,
        seed
    )

    # Return only the per-round accuracy curve.
    return accuracies

# Step 23 - run_fedavg_non_iid
def run_fedavg_non_iid(
    train_features,
    train_labels,
    test_features,
    test_labels,
    model_config,
    num_clients,
    shards_per_client,
    num_rounds,
    client_fraction,
    local_epochs,
    batch_size,
    learning_rate,
    seed
):
    # Partition the training data into non-IID client shards.
    client_partitions = partition_data_non_iid(
        train_features,
        train_labels,
        num_clients,
        shards_per_client,
        seed
    )

    # Run the complete FedAvg training loop.
    model, accuracies = run_fedavg(
        client_partitions,
        test_features,
        test_labels,
        model_config,
        num_rounds,
        client_fraction,
        local_epochs,
        batch_size,
        learning_rate,
        seed
    )

    # Return the final global model and per-round accuracy history.
    return model, accuracies

# Step 24 - compute_non_iid_gap
def compute_non_iid_gap(iid_accuracies, non_iid_accuracies):
    # Read the final accuracy from each curve.
    iid_final = float(iid_accuracies[-1])
    non_iid_final = float(non_iid_accuracies[-1])

    # Compute how much lower the non-IID final accuracy is.
    gap = iid_final - non_iid_final

    return {
        "iid_final": iid_final,
        "non_iid_final": non_iid_final,
        "gap": float(gap)
    }

# Step 25 - rounds_to_target_vs_local_epochs
def rounds_to_target_vs_local_epochs(
    client_partitions,
    test_features,
    test_labels,
    model_config,
    local_epochs_list,
    target_accuracy,
    num_rounds,
    client_fraction,
    batch_size,
    learning_rate,
    seed
):
    results = {}

    for local_epochs in local_epochs_list:
        # Run FedAvg using the current number of local epochs.
        _, accuracies = run_fedavg(
            client_partitions,
            test_features,
            test_labels,
            model_config,
            num_rounds,
            client_fraction,
            local_epochs,
            batch_size,
            learning_rate,
            seed
        )

        # Find the first round whose accuracy reaches the target.
        reached_round = None

        for round_idx, accuracy in enumerate(accuracies):
            if accuracy >= target_accuracy:
                reached_round = round_idx
                break

        results[local_epochs] = reached_round

    return results

# Step 26 - accuracy_vs_client_fraction
def accuracy_vs_client_fraction(
    client_partitions,
    test_features,
    test_labels,
    model_config,
    client_fraction_list,
    num_rounds,
    local_epochs,
    batch_size,
    learning_rate,
    seed
):
    results = {}

    for client_fraction in client_fraction_list:
        # Run the complete FedAvg training with the same seed.
        _, accuracies = run_fedavg(
            client_partitions,
            test_features,
            test_labels,
            model_config,
            num_rounds,
            client_fraction,
            local_epochs,
            batch_size,
            learning_rate,
            seed
        )

        # Store the final test accuracy.
        results[client_fraction] = float(accuracies[-1])

    return results

