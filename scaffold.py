"""
Federated Averaging (FedAvg) from Scratch in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Scaffold for Federated Averaging (FedAvg) from scratch in PyTorch.

Imports the full surface of functions the student will implement in
`solution.py`, then runs a minimal end-to-end demo on tiny synthetic data:
build data -> split -> partition -> run FedAvg rounds -> evaluate, plus a
quick non-IID vs IID comparison and a centralized baseline.
"""

import numpy as np
import torch

from solution import (
    build_mlp_classifier,
    build_synthetic_dataset,
    train_test_split_dataset,
    partition_data_iid,
    partition_data_non_iid,
    count_client_samples,
    iterate_client_batches,
    compute_batch_loss,
    local_sgd_step,
    train_client_local,
    clone_model_state,
    load_model_state,
    initialize_global_state,
    add_state_dicts,
    scale_state_dict,
    aggregate_weighted_average,
    select_round_clients,
    run_communication_round,
    evaluate_accuracy,
    run_fedavg,
    train_centralized_baseline,
    run_fedavg_iid,
    run_fedavg_non_iid,
    compute_non_iid_gap,
    rounds_to_target_vs_local_epochs,
    accuracy_vs_client_fraction,
)


def main():
    """Run a tiny FedAvg demo so the student can see the pipeline work."""
    np.random.seed(0)
    torch.manual_seed(0)

    # --- Toy experiment configuration (kept tiny for CPU) -----------------
    input_size = 8
    hidden_size = 16
    num_classes = 3
    num_samples = 300
    seed = 0

    model_config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
    }

    num_clients = 5
    num_rounds = 6
    client_fraction = 0.6
    local_epochs = 2
    batch_size = 16
    learning_rate = 0.05

    # --- Data preparation -------------------------------------------------
    features, labels = build_synthetic_dataset(
        num_samples, input_size, num_classes, seed
    )
    print(f"dataset: features={tuple(features.shape)} labels={tuple(labels.shape)}")

    train_x, train_y, test_x, test_y = train_test_split_dataset(
        features, labels, test_fraction=0.25, seed=seed
    )
    print(f"train={train_x.shape[0]} examples  test={test_x.shape[0]} examples")

    # --- Client partitioning ---------------------------------------------
    iid_parts = partition_data_iid(train_x, train_y, num_clients, seed)
    print("IID samples per client:", count_client_samples(iid_parts))

    non_iid_parts = partition_data_non_iid(
        train_x, train_y, num_clients, shards_per_client=2, seed=seed
    )
    print("non-IID samples per client:", count_client_samples(non_iid_parts))

    # --- Federated training on the IID partition --------------------------
    global_model, per_round_acc = run_fedavg(
        iid_parts, test_x, test_y, model_config,
        num_rounds=num_rounds, client_fraction=client_fraction,
        local_epochs=local_epochs, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed,
    )
    print("FedAvg per-round test accuracy:",
          [round(float(a), 3) for a in per_round_acc])
    print(f"final FedAvg accuracy: {float(evaluate_accuracy(global_model, test_x, test_y)):.3f}")

    # --- IID vs non-IID comparison ---------------------------------------
    iid_curve = run_fedavg_iid(
        train_x, train_y, test_x, test_y, model_config,
        num_clients, num_rounds, client_fraction,
        local_epochs, batch_size, learning_rate, seed,
    )
    _, non_iid_curve = run_fedavg_non_iid(
        train_x, train_y, test_x, test_y, model_config,
        num_clients, 2, num_rounds, client_fraction,
        local_epochs, batch_size, learning_rate, seed,
    )
    gap = compute_non_iid_gap(iid_curve, non_iid_curve)
    print(f"IID final={float(iid_curve[-1]):.3f}  "
          f"non-IID final={float(non_iid_curve[-1]):.3f}  gap={gap}")

    # --- Centralized baseline for reference -------------------------------
    baseline_acc = train_centralized_baseline(
        train_x, train_y, test_x, test_y, model_config,
        num_epochs=num_rounds, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed,
    )
    print(f"centralized baseline accuracy: {float(baseline_acc):.3f}")

    # --- Probes: local epochs and client fraction -------------------------
    rounds_needed = rounds_to_target_vs_local_epochs(
        iid_parts, test_x, test_y, model_config,
        local_epochs_list=[1, 2, 4], target_accuracy=0.5,
        num_rounds=num_rounds, client_fraction=client_fraction,
        batch_size=batch_size, learning_rate=learning_rate, seed=seed,
    )
    print("rounds-to-target vs local epochs:", rounds_needed)

    frac_results = accuracy_vs_client_fraction(
        iid_parts, test_x, test_y, model_config,
        client_fraction_list=[0.2, 0.6, 1.0], num_rounds=num_rounds,
        local_epochs=local_epochs, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed,
    )
    print("accuracy vs client fraction:", frac_results)


if __name__ == "__main__":
    main()
