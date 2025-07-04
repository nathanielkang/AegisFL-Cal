from .base import FedAvgStrategy
import torch
import numpy as np


class ACSFLStrategy(FedAvgStrategy):
    def __init__(self, model, clients, device, epsilon=1.0, compression_ratio_eta=0.1, num_clusters_m=1):
        """
        Initialize ACSFL strategy.
        
        Args:
            model: Global model to be optimized
            clients: List of clients
            device: Device for computation
            epsilon: Privacy budget
            compression_ratio_eta: Compression ratio (default 0.1)
            num_clusters_m: Number of clusters (default 1)
        """
        super(ACSFLStrategy, self).__init__(model, clients, device)
        self.epsilon = epsilon
        self.compression_ratio_eta = compression_ratio_eta
        self.num_clusters_m = num_clusters_m
        
        # Compute layer ranges for the model
        self._compute_model_layer_ranges()
        
    def _compute_model_layer_ranges(self):
        """
        Compute the start and end indices for each layer in the flattened model.
        """
        self.layer_ranges = {}
        current_idx = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                self.layer_ranges[name] = (current_idx, current_idx + num_params)
                current_idx += num_params
            else:
                # Non-trainable parameters (like BatchNorm running stats)
                self.layer_ranges[name] = None
                
        self.total_params = current_idx
        
    def run_round(self, round_idx, participation_rate, local_epochs, lr):
        """
        Run one round of ACSFL.
        """
        # For now, use standard FedAvg implementation
        # TODO: Implement actual ACSFL logic with compression and clustering
        return super().run_round(round_idx, participation_rate, local_epochs, lr)
        
    def _aggregate_acsfl_models(self, client_updates_info):
        """
        Aggregate models using ACSFL approach.
        """
        # For now, use standard aggregation
        # TODO: Implement actual ACSFL aggregation with compression
        return self.aggregate([{
            'state_dict': info['model_state'],
            'client': info['client']
        } for info in client_updates_info]) 