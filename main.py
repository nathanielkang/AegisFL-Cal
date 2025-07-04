import traceback

print("Debug: Starting script - before any imports")
import os
import sys
import argparse
import time
import traceback # Required for error logging
import random
import torch  # Import torch early
import timm  # Import timm for ViT models
import gc    # For memory management
from copy import deepcopy
import datetime
import json

print("Debug: Starting main.py imports")
try:
    # Import strategies
    from strategies.federated import (
        FedAvgStrategy,
        AegisFLCalStrategy,
        LDPFLStrategy,
        ACSFLStrategy,
        FedMPSStrategy,
        DPFedAvgStrategy
    )
    print("Debug: Successfully imported strategies")
    # Import AG News model correctly
    # from models.neural_networks import AGNewsNet # Moved this import to be conditional
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print("WARNING: Continuing execution despite import error. Some functionality may be limited.")

print("Debug: Continuing with utilities imports")
# Import utils
try:
    from utils.data_loader import (
        load_mnist, load_cifar10, load_cifar100, load_svhn, 
        load_celeba, load_shakespeare, load_adult, 
        load_covertype, load_credit,
        load_ag_news_direct,
        load_fashion_mnist, load_kdd_cup_99, load_newsgroups,  # Added new datasets
        # load_emnist, # Removed load_emnist
        collate_batch # Added collate_batch
    )
    print("Debug: Successfully imported data_loader")
    from utils.evaluation import evaluate_accuracy, evaluate_loss
    print("Debug: Successfully imported evaluation")
except ImportError as e:
    print(f"Debug: Error importing utilities: {e}")

print("Debug: Importing models")
try:
    from models.neural_networks import (
        MNISTNet, CIFAR10Net, CIFAR100Net, SVHNNet, CelebANet, ShakespeareLSTM, AdultNet, CovertypeNet, CreditNet,
        AGNewsNet, KDDNet, NewsGroupsNet  # Added KDDNet and NewsGroupsNet
        # EMNISTNet # Removed EMNISTNet
    )
    print("Debug: Successfully imported neural networks")
except ImportError as e:
    print(f"Debug: Error importing neural networks: {e}")

print("Debug: Continuing with remaining imports")

# Import other libraries
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import multiprocessing
from torch.utils.data import Subset # Added for text dataset client creation

# Explicitly import Client class here to ensure it's available
from data.client import Client

# Parse only the GPU argument early
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--gpu_id', type=int, default=-1, help='ID of GPU to use (-1 for all GPUs, >=0 for specific GPU)')
temp_args, _ = parser.parse_known_args()

# Set CUDA_VISIBLE_DEVICES based on argument
if torch.cuda.is_available():
    if temp_args.gpu_id == -1:
        # Use all available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            # Don't set CUDA_VISIBLE_DEVICES - this makes all GPUs visible
            print(f"*** Using all {num_gpus} available GPUs ***")
        else:
            print("*** CUDA available but no GPUs detected - using CPU ***")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # Use specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(temp_args.gpu_id)
        print(f"*** Set CUDA_VISIBLE_DEVICES to {temp_args.gpu_id} ***")
else:
    print("*** CUDA not available - will use CPU ***")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import other libraries
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import multiprocessing
from torch.utils.data import Subset # Added for text dataset client creation

# Set start method to spawn
if __name__ == '__main__':
    print("Debug: Entering if __name__ == '__main__' block")
    multiprocessing.set_start_method('spawn', force=True)

# Check if GPUs are available and print detailed info
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"*** PyTorch can see {num_gpus} GPU(s) ***")
    for i in range(num_gpus):
        print(f"*** GPU {i}: {torch.cuda.get_device_name(i)} ***")
    
    # Configure device(s) to use
    use_cuda = num_gpus > 0
    if use_cuda:
        # Determine the single device to use
        target_gpu_id = 0 # Default to GPU 0
        if temp_args.gpu_id >= 0: # User specified a valid GPU
            target_gpu_id = temp_args.gpu_id # Use the specified one (CUDA_VISIBLE_DEVICES handles mapping)
        device = torch.device(f"cuda:{target_gpu_id}")
        use_multi_gpu = False # Ensure multi-GPU is always false now
        print(f"*** Using single device: {device} (Physical GPU requested: {temp_args.gpu_id}) ***")
    else:
        device = torch.device("cpu")
        use_multi_gpu = False
        print("*** No GPU detected, using CPU instead ***")
else:
    device = torch.device("cpu")
    use_multi_gpu = False
    print("*** CUDA not available, using CPU instead ***")

try:
    from models.neural_networks import (
        MNISTNet, CIFAR10Net, CIFAR100Net, SVHNNet,
        CelebANet, ShakespeareLSTM, AdultNet, CovertypeNet, CreditNet,
        GenericVAE,
        AGNewsNet, KDDNet, NewsGroupsNet  # Added KDDNet and NewsGroupsNet
    )
    from utils.data_loader import load_dataset # This was load_dataset, ensure it's correct
    from utils.evaluation import evaluate_loss, evaluate_accuracy
    from data.client import Client # Ensure Client is imported here
    from data.server import Server
    from strategies.federated import (
        FedAvgStrategy,
        AegisFLCalStrategy,
        LDPFLStrategy,
        ACSFLStrategy,
        FedMPSStrategy,
        DPFedAvgStrategy
    )
    # Import AG News model correctly
    # from models.neural_networks import AGNewsNet # This was moved up
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print("WARNING: Continuing execution despite import error. Some functionality may be limited.")

try:
    import torchvision
    import timm
    print(f"Successfully imported timm v{timm.__version__}")
except ImportError as e:
    print(f"WARNING: Failed to import timm: {e}")
    print("You might need to install it with 'pip install timm' to use ViT models for CIFAR-100")

def get_args():
    print("Debug: Inside get_args function")
    
    full_parser = argparse.ArgumentParser(description='Federated Learning Framework')
    
    # General arguments
    full_parser.add_argument('--dataset', type=str, default='mnist', 
                             choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'ag_news', 'newsgroups', 'synthetic', 'svhn', 'adult', 'covertype', 'credit', 'kdd_cup_99'], 
                             help='Dataset to use')
    full_parser.add_argument('--data_type', type=str, default='auto',
                             choices=['auto', 'tabular', 'image', 'text'],
                             help='Type of data (auto-detect by default)')
    full_parser.add_argument('--model', type=str, default='cnn', help='Model architecture to use')
    full_parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    full_parser.add_argument('--rounds', type=int, default=10, help='Number of federated learning rounds')
    full_parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs for each client')
    full_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    full_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    full_parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer to use')
    full_parser.add_argument('--participation_rate', type=float, default=1.0, help='Fraction of clients participating in each round')
    full_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    full_parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use, -1 for CPU') # Matches early parser
    full_parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    full_parser.add_argument('--save_model', action='store_true', help='Save the trained global model')
    full_parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping in strategies that support it (e.g., AegisFLCal retraining)')
    full_parser.add_argument('--min_delta', type=float, default=0.001, help='Min delta for early stopping')

    # Data partitioning arguments
    full_parser.add_argument('--non_iid', action='store_true', help='Simulate non-IID data distribution')
    full_parser.add_argument('--partition_type', type=str, default='iid', choices=['iid', 'dirichlet', 'pathological'], 
                             help='Type of non-IID partitioning (dirichlet or pathological)')
    full_parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter for Dirichlet distribution (smaller alpha = more non-IID)')
    full_parser.add_argument('--shards_per_client', type=int, default=2, help='Number of shards per client for pathological non-IID')
    
    # Strategy arguments
    full_parser.add_argument('--strategy', type=str, default='fedavg', 
                        choices=[
                            'fedavg', 
                            'aegisflcal',
                            'ldpfl',
                            'dpfedavg',
                            'acsfl',
                            'fedmps'
                        ], 
                        help='Federated learning strategy')
    
    # AegisFL-Cal specific arguments
    full_parser.add_argument('--aegisflcal_max_s_c', type=int, default=10, help='Max projected dimension s_c for AegisFLCal')
    full_parser.add_argument('--aegisflcal_synthetic_ratio', type=float, default=1.0, help='Ratio of synthetic samples to generate in AegisFLCal')
    full_parser.add_argument('--aegisflcal_calibration_epochs', type=int, default=1, help='Number of calibration epochs for AegisFLCal')
    full_parser.add_argument('--aegisflcal_calibration_lr', type=float, default=0.0001, help='Learning rate for AegisFLCal calibration')
    full_parser.add_argument('--aegisflcal_calibration_wd', type=float, default=0.0, help='Weight decay for AegisFLCal calibration')
    full_parser.add_argument('--aegisflcal_calibration_patience', type=int, default=3, help='Early stopping patience for AegisFLCal calibration')
    full_parser.add_argument('--aegisflcal_calibration_lr_scheduler', type=str, default='none', choices=['none', 'plateau', 'cosine'], help='LR scheduler for AegisFLCal calibration')
    full_parser.add_argument('--aegisflcal_pca_variance_threshold', type=float, default=0.95, help='Variance threshold for adaptive s_c in AegisFLCal PCA')
    # Range proof arguments for AegisFL-Cal
    full_parser.add_argument('--aegisflcal_use_range_proofs', action='store_true', help='Enable range proofs for AegisFLCal')
    full_parser.add_argument('--aegisflcal_l2_bound', type=float, default=10.0, help='L2 norm bound for AegisFLCal range proofs')
    
    # LDP-FL specific arguments
    full_parser.add_argument('--ldpfl_epsilon', type=float, default=1.0, help='Epsilon for LDP-FL')
    full_parser.add_argument('--ldpfl_T_shuffling_max_delay', type=int, default=0, help='Max delay T for parameter shuffling in LDP-FL (0 to disable)')

    # ACS-FL specific arguments
    full_parser.add_argument('--acsfl_epsilon', type=float, default=1.0, help='LDP epsilon for ACS-FL')
    full_parser.add_argument('--acsfl_eta_compression_ratio', type=float, default=0.1, help='DCT compression ratio eta for ACS-FL (0.0 to 1.0)')
    full_parser.add_argument('--acsfl_num_clusters_m', type=int, default=1, help='Number of clusters for ACS-FL (simplified to 1 for now)')

    # Fed-MPS specific arguments
    full_parser.add_argument('--fedmps_epsilon', type=float, default=1.0, help='Epsilon for Fed-MPS (for accounting)')
    full_parser.add_argument('--fedmps_delta', type=float, default=1e-5, help='Delta for Fed-MPS (for accounting)')
    full_parser.add_argument('--fedmps_sigma_gaussian_noise', type=float, default=0.1, help='Standard deviation of Gaussian noise for Fed-MPS')

    # DP-FedAvg specific arguments
    full_parser.add_argument('--dpfedavg_epsilon', type=float, default=1.0, help='Privacy budget epsilon for DP-FedAvg')
    full_parser.add_argument('--dpfedavg_delta', type=float, default=1e-5, help='Privacy parameter delta for DP-FedAvg')
    full_parser.add_argument('--dpfedavg_clip_norm', type=float, default=1.0, help='Clipping norm for DP-FedAvg')
    full_parser.add_argument('--dpfedavg_noise_multiplier', type=float, default=None, help='Noise multiplier for DP-FedAvg (if None, computed from epsilon)')

    # Results dir arguments
    full_parser.add_argument('--exp_name', type=str, default=None, help='Name of experiment (used in results directory)')
    
    # Visualization arguments
    full_parser.add_argument('--visualize_clients', action='store_true', 
                            help='Visualize client data distribution before training')
    
    # Parse the arguments
    print("Debug: About to parse arguments")
    args = full_parser.parse_args()
    print("Debug: Arguments parsed successfully")
    
    return full_parser, args

def setup_device(args):
    """Set up the device to use based on args."""
    print("Debug: Inside setup_device function")
    
    if torch.cuda.is_available():
        if args.gpu_id == -1:
            # Use all available GPUs
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                device = torch.device(f"cuda:0")  # Default to first GPU
                print(f"Using device: {device}")
            else:
                device = torch.device("cpu")
                print("No GPUs detected, using CPU instead")
        else:
            # Use specific GPU
            device = torch.device(f"cuda:{args.gpu_id}")
            print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU instead")
    
    return device

def detect_data_type(dataset_name, data_type_arg):
    """Detect the type of data based on dataset name."""
    if data_type_arg != 'auto':
        return data_type_arg
    
    # Auto-detect based on dataset
    image_datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'svhn', 'celeba']
    text_datasets = ['ag_news', 'newsgroups', 'shakespeare']
    tabular_datasets = ['adult', 'covertype', 'credit', 'kdd_cup_99']
    
    if dataset_name in image_datasets:
        return 'image'
    elif dataset_name in text_datasets:
        return 'text'
    elif dataset_name in tabular_datasets:
        return 'tabular'
    else:
        print(f"Warning: Could not auto-detect data type for {dataset_name}. Defaulting to 'tabular'.")
        return 'tabular'

def process_dataset(args, device):
    """Load the dataset and create client dataloaders."""
    print("Debug: Inside process_dataset function")
    
    # Load dataset based on args.dataset
    print(f"Debug: Loading {args.dataset} dataset")
    if args.dataset == 'mnist':
        train_dataset, test_dataset = load_mnist()
    elif args.dataset == 'fashion_mnist':
        train_dataset, test_dataset = load_fashion_mnist()
    elif args.dataset == 'cifar10':
        train_dataset, test_dataset = load_cifar10()
    elif args.dataset == 'cifar100':
        train_dataset, test_dataset = load_cifar100()
    elif args.dataset == 'svhn':
        train_dataset, test_dataset = load_svhn()
    elif args.dataset == 'celeba':
        train_dataset, test_dataset = load_celeba()
    elif args.dataset == 'shakespeare':
        train_dataset, test_dataset = load_shakespeare()
    elif args.dataset == 'adult':
        train_dataset, test_dataset = load_adult()
    elif args.dataset == 'covertype':
        train_dataset, test_dataset = load_covertype()
    elif args.dataset == 'credit':
        train_dataset, test_dataset = load_credit()
    elif args.dataset == 'kdd_cup_99':
        train_dataset, test_dataset = load_kdd_cup_99()
    elif args.dataset == 'ag_news':
        train_dataset, test_dataset = load_ag_news_direct()
    elif args.dataset == 'newsgroups':
        train_dataset, test_dataset = load_newsgroups()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Debug: Dataset loaded. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Create dataloaders for testing
    # Determine if it's a text dataset needing the custom collate function for the test_loader
    base_test_dataset = test_dataset.dataset if isinstance(test_dataset, Subset) else test_dataset
    is_text_dataset_test = hasattr(base_test_dataset, 'vocab_size')
    is_shakespeare_test = hasattr(base_test_dataset, 'char_to_idx') if is_text_dataset_test else False
    use_collate_fn_test = (is_text_dataset_test and not is_shakespeare_test)
    collate_fn_for_test_loader = collate_batch if use_collate_fn_test else None
    test_batch_size = 16 if use_collate_fn_test else args.batch_size # Smaller batch for text

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn_for_test_loader)
    
    # Partition data among clients
    print(f"Debug: Partitioning data among {args.num_clients} clients using {args.partition_type} distribution with alpha={args.alpha if args.partition_type == 'dirichlet' else 'N/A'}")
    
    client_indices = [[] for _ in range(args.num_clients)]

    if args.partition_type == 'iid' or not args.non_iid:
        # Standard IID partitioning
        print("Debug: Using IID partitioning.")
        all_indices = list(range(len(train_dataset)))
        random.shuffle(all_indices)
        chunks = np.array_split(all_indices, args.num_clients)
        for i in range(args.num_clients):
            client_indices[i] = chunks[i].tolist()
    elif args.partition_type == 'dirichlet':
        print(f"Debug: Using Dirichlet partitioning with alpha={args.alpha}.")
        labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))]) # Get all labels
        num_classes = len(np.unique(labels))
        
        # Adjust min_size based on parameters
        if args.alpha <= 0.1 and args.num_clients >= 50:
            min_size = 2 # More permissive for high skew and many clients
        elif len(train_dataset) // args.num_clients < 10: # Default min_size was 10
            min_size = 1
        else:
            min_size = 10
        print(f"Debug: Dirichlet min_size per client set to: {min_size}")

        max_retries = 100  # Maximum number of re-partitioning attempts
        retry_count = 0

        while retry_count < max_retries:
            client_indices_current_attempt = [[] for _ in range(args.num_clients)]
            # Renamed idx_batch to avoid conflict if it was a global or outer scope variable
            # For safety, ensure it's a local variable for this partitioning attempt.
            idx_batch_local = [[] for _ in range(args.num_clients)] 
            
            for k_class in range(num_classes):
                idx_k_class = np.where(labels == k_class)[0]
                np.random.shuffle(idx_k_class)
                proportions = np.random.dirichlet(np.repeat(args.alpha, args.num_clients))
                
                # Proportions balancing logic (simplified version from before)
                # This part might need more sophisticated balancing for extreme cases.
                # A simple way is to ensure each client gets *some* data if possible
                # The original proportions line was: 
                # proportions = np.array([p * (len(idx_j) < (len(labels) / args.num_clients)) for p, idx_j in zip(proportions, idx_batch_local)])
                # This could lead to issues if many clients quickly exceed the average. Trying a more direct split.
                
                # Calculate current number of samples per client for proportion adjustment (less aggressive)
                current_client_samples = np.array([len(samples) for samples in idx_batch_local])
                avg_samples_per_client = len(labels) / args.num_clients
                
                # Adjust proportions to prevent clients from getting too many samples if they are already full
                # This is a heuristic and might need tuning
                adjusted_proportions = []
                for i_client in range(args.num_clients):
                    if current_client_samples[i_client] > 1.5 * avg_samples_per_client: # If client is already quite full
                        adjusted_proportions.append(proportions[i_client] * 0.1) # Strongly reduce their share of current class
                    else:
                        adjusted_proportions.append(proportions[i_client])
                proportions = np.array(adjusted_proportions)

                if proportions.sum() == 0: # Avoid division by zero if all proportions became 0
                    proportions = np.ones(args.num_clients) / args.num_clients # Fallback to equal
                else:
                    proportions = proportions / proportions.sum() # Re-normalize

                # Split indices of current class based on proportions
                proportions_cumsum = (np.cumsum(proportions) * len(idx_k_class)).astype(int)[:-1]
                class_splits = np.split(idx_k_class, proportions_cumsum)
                
                for i_client in range(args.num_clients):
                    if i_client < len(class_splits):
                        idx_batch_local[i_client].extend(class_splits[i_client].tolist())
            
            valid_partition = True
            num_empty_clients = 0
            for i_client in range(args.num_clients):
                if len(idx_batch_local[i_client]) < min_size:
                    if retry_count == max_retries -1: # On last attempt, log which clients are problematic
                         print(f"Warning (final attempt): Client {i_client} has {len(idx_batch_local[i_client])} samples (less than min_size={min_size}).")
                    valid_partition = False
                    break 
            
            if valid_partition:
                client_indices = idx_batch_local
                print(f"Debug: Dirichlet partitioning successful after {retry_count + 1} attempts.")
                break
            
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Warning: Dirichlet partitioning failed to satisfy min_size for all clients after {max_retries} attempts. Using last attempt.")
                client_indices = idx_batch_local # Use the last attempt anyway
                break
            elif retry_count % 10 == 0:
                 print(f"Warning: Dirichlet re-partitioning attempt {retry_count}/{max_retries}...")

    elif args.partition_type == 'pathological':
        # Pathological Non-IID: each client gets data from a small number of classes (e.g., 2 classes)
        print(f"Debug: Using Pathological partitioning.")
        num_shards, num_classes_per_client = 200, 2 # Example values, can be args
        if args.num_clients * num_classes_per_client > num_shards:
             num_shards = args.num_clients * num_classes_per_client
        
        shard_size = len(train_dataset) // num_shards
        labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
        data_indices = np.array(list(range(len(train_dataset))))
        
        # Sort data by label
        idx_sorted = np.argsort(labels)
        sorted_data_indices = data_indices[idx_sorted]
        
        shards = []
        for i in range(num_shards):
            shards.append(sorted_data_indices[i*shard_size : (i+1)*shard_size])
        
        random.shuffle(shards)
        shards_per_client = num_shards // args.num_clients
        
        for i in range(args.num_clients):
            assigned_shards = shards[i*shards_per_client : (i+1)*shards_per_client]
            client_indices[i] = np.concatenate(assigned_shards).tolist()
    else:
        raise ValueError(f"Unknown partition type: {args.partition_type}")

    train_loaders = []
    for i in range(args.num_clients):
        if not client_indices[i]:
            print(f"Warning: Client {i} has no data after partitioning. Skipping this client.")
            continue # Skip creating loader for client with no data
        
        client_dataset = Subset(train_dataset, client_indices[i])
        # Determine batch size based on client dataset size
        current_batch_size = min(args.batch_size, len(client_dataset)) if len(client_dataset) > 0 else 1
        if current_batch_size == 0 : current_batch_size = 1 # ensure batch_size is not 0

        # Determine if it's a text dataset needing the custom collate function
        base_train_dataset = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
        is_text_dataset_client = hasattr(base_train_dataset, 'vocab_size') 
        is_shakespeare_client = hasattr(base_train_dataset, 'char_to_idx') if is_text_dataset_client else False
        use_collate_fn_client = (is_text_dataset_client and not is_shakespeare_client)
        collate_fn_to_use = collate_batch if use_collate_fn_client else None

        client_loader = DataLoader(client_dataset, batch_size=current_batch_size, shuffle=True, collate_fn=collate_fn_to_use)
        train_loaders.append(client_loader)
    
    print(f"Debug: Created {len(train_loaders)} client dataloaders after partitioning.")
    
    return train_loaders, test_loader

def create_model(args, train_loaders, device):
    """Create the appropriate model based on the dataset."""
    print("Debug: Inside create_model function")
    
    # Choose model based on dataset
    if args.dataset == 'mnist':
        model = MNISTNet()
    elif args.dataset == 'fashion_mnist':
        model = MNISTNet()  # Fashion-MNIST has same dimensions as MNIST
    elif args.dataset == 'cifar10':
        model = CIFAR10Net()
    elif args.dataset == 'cifar100':
        model = CIFAR100Net()
    elif args.dataset == 'svhn':
        model = SVHNNet()
    elif args.dataset == 'celeba':
        model = CelebANet()
    elif args.dataset == 'shakespeare':
        # Get vocab size from dataset if available
        try:
            train_dataset = train_loaders[0].dataset
            if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'vocab_size'):
                vocab_size = train_dataset.dataset.vocab_size
            else:
                # Default vocab size
                vocab_size = 80  # Placeholder
            model = ShakespeareLSTM(vocab_size)
        except Exception as e:
            print(f"Error initializing Shakespeare model: {e}")
            vocab_size = 80  # Fallback
            model = ShakespeareLSTM(vocab_size)
    elif args.dataset == 'adult':
        model = AdultNet()
    elif args.dataset == 'covertype':
        model = CovertypeNet()
    elif args.dataset == 'credit':
        model = CreditNet()
    elif args.dataset == 'kdd_cup_99':
        model = KDDNet()  # 122 input features
    elif args.dataset == 'ag_news':
        # train_loaders[0].dataset is a CustomTextDataset, which has vocab_size and num_classes
        try:
            # Accessing the dataset through train_loaders might be problematic if train_loaders is empty
            # It's better to get it from the train_dataset returned by process_dataset directly
            # However, create_model is called with train_loaders, so we use that for now.
            # This assumes train_loaders is not empty.
            ag_train_dataset_subset = train_loaders[0].dataset 
            vocab_size = ag_train_dataset_subset.dataset.vocab_size # Access original dataset
            num_classes = ag_train_dataset_subset.dataset.num_classes # Access original dataset
            # num_classes for AG News is 4, vocab_size depends on build_vocab_from_iterator
            # Embed dim is a model hyperparameter, e.g., 100 or 300
            embed_dim = 100 # Example embed_dim, should ideally be an arg or configured
            model = AGNewsNet(vocab_size=vocab_size, embed_dim=embed_dim, num_class=num_classes)
        except Exception as e:
            print(f"Error initializing AGNewsNet. Ensure train_loaders is populated and dataset has vocab_size/num_classes. Error: {e}")
            # Fallback or re-raise, depending on desired behavior
            raise # Re-raise for now to make the issue visible
    elif args.dataset == 'newsgroups':
        # 20 Newsgroups - similar to AG News
        try:
            newsgroups_train_dataset_subset = train_loaders[0].dataset 
            vocab_size = newsgroups_train_dataset_subset.dataset.vocab_size # Access original dataset
            num_classes = newsgroups_train_dataset_subset.dataset.num_classes # Access original dataset
            embed_dim = 100 # Same as AG News
            model = NewsGroupsNet(vocab_size=vocab_size, embed_dim=embed_dim, num_class=num_classes)
        except Exception as e:
            print(f"Error initializing NewsGroupsNet. Ensure train_loaders is populated and dataset has vocab_size/num_classes. Error: {e}")
            raise
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Move to device
    model = model.to(device)
    print(f"Debug: Created model for {args.dataset} and moved to {device}")
    
    return model

def create_clients(train_loaders, args, device):
    """Create client objects for federated learning."""
    print("Debug: Inside create_clients function")
    
    clients = []
    for i, loader in enumerate(train_loaders):
        # Get the dataset from the loader
        dataset = loader.dataset
        
        # Client constructor expects dataset, not dataloader
        client = Client(
            client_id=i,
            dataset=dataset,  # Pass dataset instead of dataloader
            device=device
        )
        clients.append(client)
    
    print(f"Debug: Created {len(clients)} clients")
    return clients

def create_results_dir(args):
    """Create a directory to store results based on dataset."""
    print("Debug: Inside create_results_dir function")
    
    # Create base results directory if it doesn't exist
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # Create dataset-specific directory inside the base results directory
    dataset_specific_dir = os.path.join(args.results_dir, args.dataset)
    if not os.path.exists(dataset_specific_dir):
        os.makedirs(dataset_specific_dir)
    
    print(f"Debug: Results will be saved in directory: {dataset_specific_dir}")
    return dataset_specific_dir # Return the dataset-specific directory

def run_experiments(args, model_global, clients, test_loader, device, results_dir):
    """Run the federated learning experiments with the specified strategy."""
    print("Debug: Inside run_experiments function")
    
    # Import strategies directly here to ensure they're available
    try:
        from strategies.federated import (
            FedAvgStrategy,
            AegisFLCalStrategy,
            LDPFLStrategy,
            ACSFLStrategy,
            FedMPSStrategy,
            DPFedAvgStrategy
        )
        print("Debug: Successfully imported strategies in run_experiments")
    except ImportError as e:
        print(f"Error importing strategies: {e}")
        print("WARNING: Strategy {args.strategy} may not be available.")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"Debug: Creating {args.strategy} strategy")
    
    # Handle 'all' strategy specially
    if args.strategy == 'all':
        print("Strategy 'all' detected. Using fedavg for debugging purposes.")
        strategy = FedAvgStrategy(model_global, clients, device)
    # FedAvg is the default strategy
    elif args.strategy == 'fedavg':
        strategy = FedAvgStrategy(model_global, clients, device)
    # AegisFLCal (renamed from VOSPREA)
    elif args.strategy == 'aegisflcal':
        # Detect data type
        data_type = detect_data_type(args.dataset, args.data_type)
        print(f"Debug: Detected data type: {data_type} for dataset {args.dataset}")
        
        d_original_hint_val = None
        # is_image_dataset_for_pca_extractor = args.dataset in ['cifar10', 'cifar100', 'svhn', 'mnist', 'celeba'] # Not needed anymore

        # d_original_hint is always the dimension of features the main model expects / PCA operates on.
        if args.dataset == 'mnist':
            # MNISTNet: fc1 output is 512. Let's use that for PCA for consistency.
            d_original_hint_val = 512 
        elif args.dataset == 'fashion_mnist':
            # Fashion-MNIST uses MNISTNet: fc1 output is 512
            d_original_hint_val = 512
        elif args.dataset in ['cifar10', 'svhn']:
            # CIFAR10Net/SVHNNet: fc1 output is 64. Let's use that for PCA.
            d_original_hint_val = 64
        elif args.dataset == 'cifar100': 
            # CIFAR100Net (ResNet18 based): self.backbone.fc is replaced.
            # Input to the new self.backbone.fc sequence is num_features from original ResNet.fc (512).
            # Let's use these 512-dim features before the final classification head for PCA.
            d_original_hint_val = 512 
        elif args.dataset == 'celeba':
            # CelebANet: fc1 output is 512.
            d_original_hint_val = 512
        elif args.dataset == 'adult':
            d_original_hint_val = 14 
        elif args.dataset == 'covertype':
            d_original_hint_val = 54 
        elif args.dataset == 'credit':
            d_original_hint_val = 29 
        elif args.dataset == 'kdd_cup_99':
            d_original_hint_val = 122  # KDD Cup 99 has 122 features
        elif args.dataset == 'ag_news':
            d_original_hint_val = 100 # Text embedding dimension from AGNewsNet
        elif args.dataset == 'newsgroups':
            d_original_hint_val = 100  # Text embedding dimension from NewsGroupsNet
        else:
            print(f"Warning: d_original_hint for AegisFLCal not explicitly set for dataset {args.dataset}. Strategy will attempt to infer or use default.")

        smpc_config_for_strategy = {
            'dataset_name_for_pca_extractor': args.dataset, # Still useful for client to know dataset type
            'data_type': data_type  # Pass the detected data type
        }
        zkp_config_for_strategy = {
            # Adaptive ZKP approach - no manual parameters needed
            'use_adaptive_bounds': True,  # Flag to indicate we're using adaptive approach
            'use_range_proofs': args.aegisflcal_use_range_proofs,  # Enable range proofs if specified
            'l2_bound': args.aegisflcal_l2_bound  # L2 norm bound for range proofs
        }

        strategy = AegisFLCalStrategy(
            model_global, clients, device,
            subspace_dim=args.aegisflcal_max_s_c,  # Updated parameter name
            smpc_pca_config=smpc_config_for_strategy, 
            he_scheme_config={}, # Add placeholder for HE config if needed in future
            zkp_config=zkp_config_for_strategy,
            synthetic_ratio=args.aegisflcal_synthetic_ratio,
            calibration_epochs=args.aegisflcal_calibration_epochs,  # Updated parameter name
            calibration_lr=args.aegisflcal_calibration_lr,  # Updated parameter name
            calibration_wd=args.aegisflcal_calibration_wd,  # Updated parameter name
            calibration_patience=args.aegisflcal_calibration_patience,  # Updated parameter name
            calibration_lr_scheduler=args.aegisflcal_calibration_lr_scheduler,  # Updated parameter name
            d_original_hint=d_original_hint_val,
            pca_variance_explained_threshold=args.aegisflcal_pca_variance_threshold,
            data_type=data_type  # Pass data type to strategy
        )
    # LDP-FL
    elif args.strategy == 'ldpfl':
        strategy = LDPFLStrategy(
            model_global, clients, device,
            ldpfl_epsilon=args.ldpfl_epsilon,
            ldpfl_T_shuffling_max_delay=args.ldpfl_T_shuffling_max_delay
        )
    # ACS-FL (replaces FedLAPDP)
    elif args.strategy == 'acsfl': 
        strategy = ACSFLStrategy(
            model_global, clients, device,
            epsilon=args.acsfl_epsilon,
            compression_ratio_eta=args.acsfl_eta_compression_ratio,
            num_clusters_m=args.acsfl_num_clusters_m
        )
    elif args.strategy == 'fedmps':
        strategy = FedMPSStrategy(
            model_global, clients, device,
            fedmps_epsilon=args.fedmps_epsilon,
            fedmps_delta=args.fedmps_delta,
            fedmps_sigma_gaussian_noise=args.fedmps_sigma_gaussian_noise
        )
    elif args.strategy == 'dpfedavg':
        strategy = DPFedAvgStrategy(
            model_global, clients, device,
            epsilon=args.dpfedavg_epsilon,
            delta=args.dpfedavg_delta,
            clip_norm=args.dpfedavg_clip_norm,
            noise_multiplier=args.dpfedavg_noise_multiplier
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    print(f"Debug: Running {args.rounds} rounds of training")
    
    # Placeholder for results
    results = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_per_class_accuracy': [],  # New key for per-class accuracy dictionaries
        'round_time_s': [] # Added for round execution time
    }
    
    # Evaluate initial model
    print("Debug: Evaluating initial model")
    try:
        test_loss = evaluate_loss(model_global, test_loader, device)
        # Unpack the tuple returned by evaluate_accuracy
        test_accuracy, per_class_acc_dict = evaluate_accuracy(model_global, test_loader, device)
        results['train_loss'].append(0)  # Placeholder for initial train loss
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_accuracy)
        results['test_per_class_accuracy'].append(per_class_acc_dict)  # Store per-class accuracies
        results['round_time_s'].append(0.0) # No training time for initial eval
        print(f"Initial Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        # Optionally print per-class accuracies for targeted datasets
        if args.dataset in ['cifar10', 'svhn', 'ag_news', 'kdd_cup_99']:
            print(f"Initial Per-Class Accuracies: {per_class_acc_dict}")
    except Exception as e:
        print(f"Error in initial evaluation: {e}")
        traceback.print_exc()
    
    # For debugging, we'll limit to just 1 round
    # debug_rounds = min(args.rounds, 1) # Commenting out to run full rounds
    # print(f"Debug: Running {debug_rounds} rounds for debugging")
    
    # Run training for specified number of rounds
    print(f"Debug: Running {args.rounds} rounds as per arguments.")
    for round_idx in range(args.rounds):
        print(f"Debug: Round {round_idx+1}/{args.rounds}")
        try:
            round_start_time = time.time()
            # Run one round of training
            # Strategy.run_round might return (state_dict, loss) or just loss
            # Adjusting based on FedMPSStrategy returning (state_dict, avg_train_loss)
            round_result = strategy.run_round(round_idx, args.participation_rate, args.local_epochs, args.lr)
            
            # Unpack based on what strategy returns
            if isinstance(round_result, tuple) and len(round_result) == 2:
                # Assuming the second element is the scalar loss value we need
                # The first element (e.g., model state dict) is handled by the strategy internally
                train_loss = round_result[1] 
            else:
                # Fallback if only a scalar loss is returned (like some older strategies might)
                train_loss = round_result 

            round_end_time = time.time()
            round_duration = round_end_time - round_start_time
            
            # Evaluate
            test_loss = evaluate_loss(model_global, test_loader, device)
            # Unpack the tuple returned by evaluate_accuracy
            test_accuracy, per_class_acc_dict = evaluate_accuracy(model_global, test_loader, device)
            
            # Record results
            results['train_loss'].append(train_loss)
            results['test_loss'].append(test_loss)
            results['test_accuracy'].append(test_accuracy)
            results['test_per_class_accuracy'].append(per_class_acc_dict)  # Store per-class accuracies
            results['round_time_s'].append(round_duration)
            
            print(f"Round {round_idx+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Time: {round_duration:.2f}s")
            # Optionally print per-class accuracies for targeted datasets
            if args.dataset in ['cifar10', 'svhn', 'ag_news', 'kdd_cup_99']:
                print(f"Round {round_idx+1} Per-Class Accuracies: {per_class_acc_dict}")
        except Exception as e:
            import traceback  # Import traceback locally in the exception handler
            print(f"Error in round {round_idx+1}: {e}")
            traceback.print_exc()
    
    # Save results
    print("Debug: Saving results")
    try:
        # New filename format: dataset_strategy_rounds<R>_clients<N>_alpha<A>.csv
        alpha_str = str(args.alpha).replace('.', '_') if args.non_iid and args.partition_type == 'dirichlet' else ('path' if args.non_iid and args.partition_type == 'pathological' else 'iid')
        results_filename = f"{args.dataset}_{args.strategy}_rounds{args.rounds}_clients{args.num_clients}_alpha{alpha_str}.csv"
        results_file_path = os.path.join(results_dir, results_filename)
        
        with open(results_file_path, 'w') as f:
            # Write CSV header with dynamic per-class accuracy columns for target datasets
            header_cols = ["round", "train_loss", "test_loss", "test_accuracy", "round_time_s"]
            num_classes_for_csv = 0
            if args.dataset in ['cifar10', 'svhn']:
                num_classes_for_csv = 10  # Both CIFAR-10 and SVHN have 10 classes
                for c in range(num_classes_for_csv):
                    header_cols.append(f"class_{c}_acc")
            elif args.dataset == 'ag_news':
                num_classes_for_csv = 4  # AG News has 4 classes
                for c in range(num_classes_for_csv):
                    header_cols.append(f"class_{c}_acc")
            elif args.dataset == 'kdd_cup_99':
                num_classes_for_csv = 5  # KDD Cup 99 has 5 classes (Normal, DOS, R2L, U2R, Probe)
                for c in range(num_classes_for_csv):
                    header_cols.append(f"class_{c}_acc")
            f.write(",".join(header_cols) + "\n")
            
            # Write data rows
            # First row is for initial evaluation (round 0)
            initial_eval_data = [
                0,
                results['train_loss'][0],
                results['test_loss'][0],
                results['test_accuracy'][0],
                results['round_time_s'][0]
            ]
            if num_classes_for_csv > 0:
                initial_per_class_acc = results['test_per_class_accuracy'][0]
                for c in range(num_classes_for_csv):
                    initial_eval_data.append(initial_per_class_acc.get(c, float('nan')))  # Use NaN if class missing
            f.write(",".join(map(str, initial_eval_data)) + "\n")
            
            # Remaining rows for actual training rounds
            for round_idx_csv in range(len(results['train_loss'])-1):
                round_num_csv = round_idx_csv + 1
                current_round_data = [
                    round_num_csv,
                    results['train_loss'][round_num_csv],
                    results['test_loss'][round_num_csv],
                    results['test_accuracy'][round_num_csv],
                    results['round_time_s'][round_num_csv]
                ]
                if num_classes_for_csv > 0:
                    per_class_acc_this_round = results['test_per_class_accuracy'][round_num_csv]
                    for c in range(num_classes_for_csv):
                        current_round_data.append(per_class_acc_this_round.get(c, float('nan')))
                f.write(",".join(map(str, current_round_data)) + "\n")
        
        print(f"Results saved to {results_file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        traceback.print_exc()
    
    return results

def main():
    try:
        print("Debug: Starting __main__ block")
        
        # The rest of the code
        # Get args
        try:
            print("Debug: About to get arguments")
            parser, args = get_args()
            print(f"Debug: Parsed args - dataset: {args.dataset}, strategy: {args.strategy}")
        except Exception as e:
            import traceback  # Import locally to avoid the previous error
            print(f"Debug: Error in get_args(): {e}")
            traceback.print_exc()
            return
        
        try:
            # Set random seed
            seed = args.seed
            print(f"Debug: Setting random seed: {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)
        except Exception as e:
            import traceback  # Import locally in each exception handler
            print(f"Debug: Error setting random seed: {e}")
            traceback.print_exc()
            return
        
        try:
            # 0. Device setup
            device = setup_device(args)
            print(f"Debug: Device setup complete: {device}")
        except Exception as e:
            import traceback
            print(f"Debug: Error in device setup: {e}")
            traceback.print_exc()
            return
        
        try:
            # 1. Load dataset
            print(f"Debug: About to load dataset: {args.dataset}")
            train_loaders, test_loader = process_dataset(args, device)
            if train_loaders and test_loader:
                print(f"Debug: Dataset loaded. Number of clients: {len(train_loaders)}")
                
                # Visualize client data distribution if requested
                if hasattr(args, 'visualize_clients') and args.visualize_clients:
                    from visualization import visualize_client_distribution_histogram, visualize_client_class_distribution
                    
                    # Create visualization directory
                    viz_dir = os.path.join(args.results_dir, 'client_distributions')
                    if not os.path.exists(viz_dir):
                        os.makedirs(viz_dir)
                    
                    # Generate histograms
                    hist_path = os.path.join(viz_dir, f'{args.dataset}_client_distribution.png')
                    visualize_client_distribution_histogram(train_loaders, args.dataset, save_path=hist_path)
                    
                    # Generate pie charts for sample clients
                    pie_path = os.path.join(viz_dir, f'{args.dataset}_client_classes.png')
                    visualize_client_class_distribution(train_loaders, args.dataset, num_clients_to_show=10, save_path=pie_path)
                    
                    print(f"Client distribution visualizations saved to {viz_dir}")
        except Exception as e:
            import traceback
            print(f"Debug: Error loading dataset: {e}")
            traceback.print_exc()
            return
        
        try:
            # 2. Create model
            print(f"Debug: About to create model for dataset: {args.dataset}")
            model_global = create_model(args, train_loaders, device)
            if model_global:
                print(f"Debug: Model created: {type(model_global).__name__}")
        except Exception as e:
            import traceback
            print(f"Debug: Error creating model: {e}")
            traceback.print_exc()
            return
        
        try:
            # 3. Create clients
            print(f"Debug: About to create clients")
            clients = create_clients(train_loaders, args, device)
            print(f"Debug: Created {len(clients)} clients")
        except Exception as e:
            import traceback
            print(f"Debug: Error creating clients: {e}")
            traceback.print_exc()
            return
        
        try:
            # 4. Set up results directory
            print(f"Debug: Setting up results directory")
            results_dir = create_results_dir(args)
            print(f"Debug: Results directory: {results_dir}")
        except Exception as e:
            import traceback
            print(f"Debug: Error setting up results directory: {e}")
            traceback.print_exc()
            return
        
        try:
            # 5. Run the experiment
            print(f"Debug: About to run experiments with strategy: {args.strategy}")
            model_results = run_experiments(args, model_global, clients, test_loader, device, results_dir)
        except Exception as e:
            import traceback
            print(f"Debug: Error running experiments: {e}")
            traceback.print_exc()
            return
        
        print("Debug: Experiment completed successfully")
    except Exception as e:
        import traceback
        print(f"Debug: Error in main execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Debug: Entering if __name__ == '__main__' block")
    main() 