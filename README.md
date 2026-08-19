# Federated Averaging (FedAvg) from Scratch in PyTorch

Implement the FedAvg algorithm end to end, from client data partitioning and local SGD training to weighted state-dict aggregation across communication rounds. Then probe how non-IID data, local epochs, and client participation shape federated learning performance.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** build_mlp_classifier
- [x] **2.** build_synthetic_dataset
- [x] **3.** train_test_split_dataset
- [x] **4.** partition_data_iid
- [x] **5.** partition_data_non_iid
- [x] **6.** count_client_samples
- [x] **7.** iterate_client_batches
- [x] **8.** compute_batch_loss
- [x] **9.** local_sgd_step
- [x] **10.** train_client_local
- [x] **11.** clone_model_state
- [x] **12.** load_model_state
- [x] **13.** initialize_global_state
- [x] **14.** add_state_dicts
- [x] **15.** scale_state_dict
- [x] **16.** aggregate_weighted_average
- [x] **17.** select_round_clients
- [x] **18.** run_communication_round
- [x] **19.** evaluate_accuracy
- [x] **20.** run_fedavg
- [x] **21.** train_centralized_baseline
- [x] **22.** run_fedavg_iid
- [x] **23.** run_fedavg_non_iid
- [x] **24.** compute_non_iid_gap
- [x] **25.** rounds_to_target_vs_local_epochs
- [x] **26.** accuracy_vs_client_fraction

## Results

```
dataset: features=(300, 8) labels=(300,)
train=225 examples  test=75 examples
IID samples per client: [45, 45, 45, 45, 45]
non-IID samples per client: [46, 44, 45, 45, 45]
FedAvg per-round test accuracy: [0.427, 0.413, 0.4, 0.413, 0.427, 0.373]
final FedAvg accuracy: 0.373
IID final=0.373  non-IID final=0.373  gap={'iid_final': 0.3733333349227905, 'non_iid_final': 0.3733333349227905, 'gap': 0.0}
centralized baseline accuracy: 0.440
rounds-to-target vs local epochs: {1: None, 2: None, 4: None}
accuracy vs client fraction: {0.2: 0.41333332657814026, 0.6: 0.3733333349227905, 1.0: 0.4000000059604645}
```
