@'
# Evaluating Differential Privacy and Secure Aggregation Against Inference Attacks in Federated Learning for Autonomous Vehicles

## Overview

This project evaluates privacy-preserving techniques in Federated Learning (FL) using the VeReMi vehicular network dataset.

The focus is on understanding the trade-off between model utility and privacy when applying Differential Privacy (DP) and Secure Aggregation (SecAgg), and evaluating the exposure of model updates to inference and reconstruction attacks.

## Objectives

- Implement Federated Learning using Flower and PyTorch.
- Establish a baseline using Federated Averaging (FedAvg).
- Evaluate class-weighted and unweighted training.
- Apply Differential Privacy using gradient clipping and noise.
- Implement Secure Aggregation to prevent the server from observing individual client updates.
- Evaluate Deep Leakage from Gradients (DLG) and model-update reconstruction.
- Compare predictive utility across privacy configurations.

## Technology Stack

- Python
- PyTorch
- Flower
- Opacus
- NumPy
- Pandas
- VeReMi dataset

## Dataset and Experimental Setup

The project uses the VeReMi vehicular network dataset.

The processed dataset contains 12 input features and a binary target.

A client-disjoint data split is used to prevent the same clients from appearing in training and held-out testing.

| Partition | Records | Clients |
|---|---:|---:|
| Training | 1,472,198 | 3,388 |
| Validation | 1,450,978 | 3,338 |
| Held-out test | 386,815 | 847 |

The federated training experiments use selected training clients. Validation and test clients are kept separate from the selected training clients.

The original dataset is not included in this repository. Obtain the VeReMi dataset separately and prepare it using the project's data-processing scripts.

## Privacy Mechanisms

### Differential Privacy

Differential Privacy is applied during client-side training using gradient clipping and Gaussian noise through Opacus.

The objective is to limit the information that individual training records can reveal through model updates.

### Secure Aggregation

Flower's Secure Aggregation Plus workflow is used to mask individual client updates.

The server aggregates protected updates without directly observing each participating client's individual update.

### Combined DP + Secure Aggregation

The combined experiment applies client-side Differential Privacy together with Secure Aggregation.

These mechanisms provide different protections:
- Differential Privacy limits information leakage about individual training records.
- Secure Aggregation protects individual client updates during aggregation.

## Inference and Reconstruction Attack Evaluation

The repository includes experiments and utilities for:
- Deep Leakage from Gradients (DLG).
- Model-update inversion and reconstruction.
- Comparing reconstruction errors under different privacy configurations.

The attack evaluation is intended to examine how privacy mechanisms affect the information exposed through gradients and model updates.

## Experimental Results

The following results were obtained on the held-out, client-disjoint test partition. Classification thresholds were selected using validation data.

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline FedAvg | 0.6817 | 0.3917 | 0.4014 | 0.3965 |
| Class-weighted FedAvg | 0.7111 | 0.4668 | 0.4788 | 0.4727 |
| DP-FedAvg (weighted) | 0.5187 | 0.2012 | 0.9999 | 0.3351 |
| DP-FedAvg (unweighted) | 0.4858 | 0.2077 | 0.9258 | 0.3393 |
| DP + Secure Aggregation | 0.5042 | 0.5914 | 0.2563 | 0.3576 |

### Observations

- Class-weighted FedAvg achieved the best predictive performance among the evaluated models.
- Differential Privacy introduced a substantial utility trade-off in the current experiments.
- The combined DP + Secure Aggregation model achieved higher precision but lower recall.
- The DP + Secure Aggregation model's ROC-AUC was close to random ranking, indicating that further optimization is needed.
- Secure Aggregation and Differential Privacy address different privacy threats and should not be treated as interchangeable.

These results are preliminary experimental findings, not a claim that the combined privacy configuration outperforms non-private training.

## Repository Structure

```text
fl-project/
├── fl_app/                 # Flower client and server applications
├── src/                    # Training, evaluation, privacy and attack modules
├── checkpoints/            # Saved model checkpoints
├── results/                # Experiment metrics and attack results
├── config.py               # Project configuration
├── main.py                 # Main project entry point
├── pyproject.toml           # Project and Flower configuration
├── requirements.txt         # Python dependencies
└── README.md