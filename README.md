# AegisFL-Cal: A Federated Learning Framework with Privacy-Preserving Calibration

This repository contains the implementation of **AegisFL-Cal**, a federated learning framework that enhances privacy preservation through cryptographic techniques while maintaining model accuracy via calibration mechanisms. This code accompanies our IEEE journal paper submission.

## Overview

The framework implements multiple federated learning strategies including:
- **AegisFL-Cal**: Our proposed method with privacy-preserving calibration
- **FedAvg**: Standard federated averaging baseline
- **FedProx**: Federated optimization with proximal term
- **MOON**: Model-contrastive federated learning
- **LDP-FL**: Local differential privacy federated learning
- **DP-FedAvg**: Differentially private federated averaging
- **ACS-FL**: Anonymous communication system for federated learning
- **Fed-MPS**: Federated learning with multi-party secure aggregation
- **SMPC-FedAvg**: Secure multi-party computation federated averaging

## System Requirements

- Python 3.8+
- PyTorch 1.9.0+
- CUDA compatible GPU (optional but recommended)
- TenSEAL library for homomorphic encryption (required for AegisFL-Cal)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. For AegisFL-Cal cryptographic features:
```bash
pip install tenseal
```

## Dataset Support

The framework supports multiple datasets across different domains:

### Image Datasets
- MNIST
- Fashion-MNIST  
- CIFAR-10
- CIFAR-100
- SVHN

### Text Datasets
- AG News
- 20 Newsgroups

### Tabular Datasets
- Adult
- Covertype
- Credit
- KDD Cup 99

## Quick Start

### Basic Usage

Run AegisFL-Cal on MNIST with 10 clients:
```bash
python main.py --dataset mnist --num_clients 10 --rounds 20 --strategy aegisflcal
```

Run FedAvg baseline for comparison:
```bash
python main.py --dataset mnist --num_clients 10 --rounds 20 --strategy fedavg
```

### Advanced Configuration

#### Non-IID Data Distribution
```bash
python main.py --dataset cifar10 --num_clients 100 --rounds 50 \
    --strategy aegisflcal --non_iid --partition_type dirichlet --alpha 0.1
```

#### AegisFL-Cal Specific Parameters
```bash
python main.py --dataset ag_news --num_clients 50 --rounds 30 \
    --strategy aegisflcal \
    --aegisflcal_max_s_c 15 \
    --aegisflcal_synthetic_ratio 1.5 \
    --aegisflcal_calibration_epochs 3 \
    --aegisflcal_calibration_lr 0.0001 \
    --aegisflcal_use_range_proofs \
    --aegisflcal_l2_bound 10.0
```

#### Privacy-Preserving Baselines
```bash
# LDP-FL
python main.py --dataset mnist --strategy ldpfl --ldpfl_epsilon 1.0

# DP-FedAvg  
python main.py --dataset mnist --strategy dpfedavg --dpfedavg_epsilon 1.0 --dpfedavg_clip_norm 1.0

# ACS-FL
python main.py --dataset mnist --strategy acsfl --acsfl_epsilon 1.0 --acsfl_eta_compression_ratio 0.1

# Fed-MPS
python main.py --dataset mnist --strategy fedmps --fedmps_epsilon 1.0 --fedmps_sigma_gaussian_noise 0.1
```

## Key Parameters

### General Parameters
- `--dataset`: Dataset to use (mnist, cifar10, ag_news, etc.)
- `--num_clients`: Number of participating clients
- `--rounds`: Number of federated learning rounds
- `--local_epochs`: Local training epochs per client
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--participation_rate`: Fraction of clients participating per round

### Data Distribution
- `--non_iid`: Enable non-IID data distribution
- `--partition_type`: Type of partitioning (iid, dirichlet, pathological)
- `--alpha`: Dirichlet distribution parameter (smaller = more non-IID)

### AegisFL-Cal Specific
- `--aegisflcal_max_s_c`: Maximum projected dimension
- `--aegisflcal_synthetic_ratio`: Synthetic data generation ratio
- `--aegisflcal_calibration_epochs`: Calibration training epochs
- `--aegisflcal_calibration_lr`: Calibration learning rate
- `--aegisflcal_use_range_proofs`: Enable cryptographic range proofs
- `--aegisflcal_l2_bound`: L2 norm bound for range proofs

## Project Structure

```
├── main.py                           # Main entry point
├── requirements.txt                  # Dependencies
├── strategies/
│   └── federated/                   # Modular FL strategy implementations
│       ├── __init__.py              # Package initialization
│       ├── base.py                  # FedAvg base strategy
│       ├── fedprox.py               # FedProx strategy
│       ├── moon.py                  # MOON strategy
│       ├── aegisfl_cal.py           # AegisFL-Cal strategy
│       ├── ldpfl.py                 # LDP-FL strategy
│       ├── dpfedavg.py              # DP-FedAvg strategy
│       ├── acsfl.py                 # ACS-FL strategy
│       ├── fedmps.py                # Fed-MPS strategy
│       └── smpc_fedavg.py           # SMPC-FedAvg strategy
├── models/
│   └── neural_networks.py           # Model architectures
├── utils/
│   ├── data_loader.py               # Dataset loading utilities
│   ├── evaluation.py                # Model evaluation functions
│   ├── crypto_real.py               # Cryptographic implementations
│   └── synthetic_generation.py      # Synthetic data generation
├── data/
│   ├── client.py                    # Client implementation
│   ├── server.py                    # Server implementation
│   └── [dataset_folders]/           # Dataset storage
└── README.md                        # This file
```

## Recent Updates

- **Modular Architecture**: Refactored strategies into separate modules for better maintainability
- **Enhanced Strategy Support**: Added FedProx and MOON strategies
- **Improved Error Handling**: Fixed indentation issues in AegisFL-Cal implementation
- **Better Code Organization**: Each strategy now has its own file with clear interfaces

## Experimental Results

The framework has been evaluated on multiple datasets with various privacy and non-IID settings. Key findings include:

1. **Privacy-Accuracy Trade-off**: AegisFL-Cal achieves better privacy-accuracy trade-offs compared to baseline methods
2. **Non-IID Robustness**: The calibration mechanism enhances performance under heterogeneous data distributions
3. **Scalability**: Efficient performance with up to 300+ clients
4. **Cryptographic Security**: Formal privacy guarantees through homomorphic encryption and zero-knowledge proofs



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about the implementation or paper, please contact:
- NATHANIEL KANG - natekang@yonsei.ac.kr


