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
- [ ] **8.** compute_batch_loss
- [ ] **9.** local_sgd_step
- [ ] **10.** train_client_local
- [ ] **11.** clone_model_state
- [ ] **12.** load_model_state
- [ ] **13.** initialize_global_state
- [ ] **14.** add_state_dicts
- [ ] **15.** scale_state_dict
- [ ] **16.** aggregate_weighted_average
- [ ] **17.** select_round_clients
- [ ] **18.** run_communication_round
- [ ] **19.** evaluate_accuracy
- [ ] **20.** run_fedavg
- [ ] **21.** train_centralized_baseline
- [ ] **22.** run_fedavg_iid
- [ ] **23.** run_fedavg_non_iid
- [ ] **24.** compute_non_iid_gap
- [ ] **25.** rounds_to_target_vs_local_epochs
- [ ] **26.** accuracy_vs_client_fraction

---

Built on Deep-ML.
