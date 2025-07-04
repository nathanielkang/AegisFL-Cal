import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from copy import deepcopy
from collections import defaultdict, OrderedDict
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import Subset, TensorDataset, DataLoader, random_split
from typing import List, Dict, Any, Union


class FedAvgStrategy:
    def __init__(self, model, clients, device):
        """
        Initialize FedAvg strategy.
        
        Args:
            model: Global model to be optimized
            clients: List of clients
            device: Device for computation
        """
        self.model = model
        self.clients = clients
        self.device = device
        self.P_y = {}  # Global label distribution
        
    def select_clients(self, num_clients):
        """
        Randomly select clients for training.
        
        Args:
            num_clients: Number of clients to select
            
        Returns:
            selected_clients: List of selected clients
        """
        return random.sample(self.clients, min(num_clients, len(self.clients)))
    
    def aggregate(self, client_updates):
        """
        Aggregate updates from clients using weighted average based on data size (n_k).
        
        Args:
            client_updates: List of dictionaries, each containing 'state_dict' and 'client'.
            
        Returns:
            global_state: Aggregated state dict
        """
        # Check if model is DataParallel and handle accordingly
        is_data_parallel = isinstance(self.model, nn.DataParallel)
        
        # Create a deep copy of the first client's state
        global_state = {}
        
        # Initialize with zeros based on the global model's state dict keys
        if not client_updates:
            return self.model.state_dict() # Return current model if no updates

        # Use global model keys for initialization
        global_model_state = self.model.state_dict()
        for key in global_model_state.keys():
            # No need to check for 'module.' prefix here, use the canonical key
            global_state[key] = torch.zeros_like(global_model_state[key])
        
        # Calculate weights (based on data size n_k for FedAvg)
        total_samples = 0
        client_weights = []
        for update in client_updates:
            client = update['client']
            _, n_k = client.compute_local_label_distribution()
            client_weights.append(n_k)
            total_samples += n_k

        if total_samples > 0:
            normalized_weights = [w / total_samples for w in client_weights]
        else:
            normalized_weights = [1.0 / len(client_updates) for _ in client_updates]

        # Sum the updates with weights
        for i, update in enumerate(client_updates):
            state_dict = update['state_dict']
            weight = normalized_weights[i]
            for key in state_dict.keys():
                if is_data_parallel and not key.startswith('module.'):
                    global_key = 'module.' + key
                else:
                    global_key = key
                    
                # Ensure tensors are on the same device
                if state_dict[key].device != global_state[global_key].device:
                    state_dict[key] = state_dict[key].to(global_state[global_key].device)
                    
                # Check if key exists in global_state (it should based on initialization)
                if global_key in global_state:
                    if global_state[global_key].is_floating_point():
                        global_state[global_key] += weight * state_dict[key]
                    # else:
                    #    # Optionally, handle non-float tensors differently, e.g., sum 'num_batches_tracked'
                    #    # For now, we simply skip non-float tensors for averaging
                    #    pass 
                else:
                    # This case should ideally not happen if initialization is correct
                    # Handle potential key mismatches if necessary (e.g., for DataParallel)
                    print(f"Warning: Key {global_key} not found during FedAvg aggregation.")

        
        return global_state

    def run_round(self, round_idx, participation_rate, local_epochs, lr):
        """
        Run one round of Federated Averaging.

        Args:
            round_idx: Current round index.
            participation_rate: Fraction of clients to select for this round.
            local_epochs: Number of local epochs for client training.
            lr: Learning rate for client training.

        Returns:
            avg_round_loss: Average training loss for the round (placeholder).
        """
        # 1. Select clients for this round
        num_selected_clients = max(1, int(participation_rate * len(self.clients)))
        selected_clients = self.select_clients(num_selected_clients)
        # print(f"FedAvg Round {round_idx+1}: Selected {len(selected_clients)} clients")

        client_updates = []
        round_losses = [] # List to store losses from clients

        for client in selected_clients:
            # Client performs local update. 
            # The client.local_update method is expected to handle model copying and training.
            # It should return (state_dict_of_updated_local_model, avg_training_loss_for_client).
            # We pass a deepcopy of the current global model to the client to avoid in-place modification issues.
            local_model_state_dict, client_loss = client.local_update(deepcopy(self.model), 0.0, local_epochs, lr)
            
            client_updates.append({
                'state_dict': local_model_state_dict,
                'client': client # Needed for weighted aggregation by n_k
            })
            if client_loss is not None and not (isinstance(client_loss, float) and np.isnan(client_loss)):
                round_losses.append(client_loss)

        if not client_updates:
            # print(f"FedAvg Round {round_idx+1}: No client updates received.")
            return 0.0 # No loss if no updates, or handle as NaN appropriately elsewhere

        # 3. Aggregate updates
        # print(f"FedAvg Round {round_idx+1}: Aggregating {len(client_updates)} client updates.")
        aggregated_state_dict = self.aggregate(client_updates)

        # 4. Update global model
        self.model.load_state_dict(aggregated_state_dict)
        # print(f"FedAvg Round {round_idx+1}: Global model updated.")

        # Calculate average round loss
        if round_losses:
            avg_round_loss = sum(round_losses) / len(round_losses)
        else:
            avg_round_loss = float('nan') # Indicate that no valid losses were reported
        return avg_round_loss 