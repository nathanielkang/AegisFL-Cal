from .base import FedAvgStrategy
import torch
import numpy as np
from copy import deepcopy


class FedMPSStrategy(FedAvgStrategy):
    def __init__(self, model, clients, device, 
                 fedmps_epsilon, fedmps_delta, fedmps_sigma_gaussian_noise):
        """
        Initialize FedMPS strategy.
        
        Args:
            model: Global model to be optimized
            clients: List of clients
            device: Device for computation
            fedmps_epsilon: Privacy budget epsilon
            fedmps_delta: Privacy parameter delta
            fedmps_sigma_gaussian_noise: Gaussian noise sigma
        """
        super(FedMPSStrategy, self).__init__(model, clients, device)
        self.fedmps_epsilon = fedmps_epsilon
        self.fedmps_delta = fedmps_delta
        self.fedmps_sigma_gaussian_noise = fedmps_sigma_gaussian_noise
        
    def aggregate_selected_deltas(self, client_selected_deltas_info):
        """
        Aggregate selected deltas from clients.
        """
        # For now, return empty dict
        # TODO: Implement actual FedMPS aggregation logic
        return {}
        
    def run_round(self, round_idx, participation_rate, local_epochs, lr):
        """
        Run one round of FedMPS.
        """
        num_selected_clients = max(1, int(participation_rate * len(self.clients)))
        selected_clients = self.select_clients(num_selected_clients)
        
        client_updates = []
        round_losses = []
        
        for client in selected_clients:
            # Standard local update
            local_model_state_dict, client_loss = client.local_update(deepcopy(self.model), 0.0, local_epochs, lr)
            
            client_updates.append({
                'state_dict': local_model_state_dict,
                'client': client
            })
            if client_loss is not None and not (isinstance(client_loss, float) and np.isnan(client_loss)):
                round_losses.append(client_loss)
        
        if not client_updates:
            return 0.0
        
        # Standard aggregation for now
        # TODO: Implement actual FedMPS logic with delta selection and privacy
        aggregated_state_dict = self.aggregate(client_updates)
        self.model.load_state_dict(aggregated_state_dict)
        
        avg_round_loss = sum(round_losses) / len(round_losses) if round_losses else float('nan')
        return avg_round_loss 