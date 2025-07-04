from .base import FedAvgStrategy
import torch
import numpy as np
from copy import deepcopy


class DPFedAvgStrategy(FedAvgStrategy):
    def __init__(self, model, clients, device, epsilon=1.0, delta=1e-5, clip_norm=1.0, noise_multiplier=None):
        """
        Initialize DPFedAvg strategy with differential privacy.
        
        Args:
            model: Global model to be optimized
            clients: List of clients
            device: Device for computation
            epsilon: Privacy budget
            delta: Privacy parameter delta
            clip_norm: Gradient clipping norm
            noise_multiplier: Noise multiplier for DP
        """
        super(DPFedAvgStrategy, self).__init__(model, clients, device)
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        
        # Calculate noise multiplier if not provided
        if noise_multiplier is None:
            # Simple approximation for noise multiplier
            self.noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        else:
            self.noise_multiplier = noise_multiplier
            
    def clip_model_update(self, update_dict, clip_norm):
        """
        Clip model update to ensure bounded sensitivity.
        """
        # Calculate the global norm of the update
        total_norm = 0.0
        for key, value in update_dict.items():
            if torch.is_tensor(value) and value.is_floating_point():
                param_norm = value.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip the update
        clip_coef = clip_norm / (total_norm + 1e-6)
        clip_coef_clamped = min(clip_coef, 1.0)
        
        clipped_update = {}
        for key, value in update_dict.items():
            if torch.is_tensor(value) and value.is_floating_point():
                clipped_update[key] = value * clip_coef_clamped
            else:
                clipped_update[key] = value
                
        return clipped_update
        
    def add_gaussian_noise(self, aggregated_update, clip_norm, noise_multiplier):
        """
        Add Gaussian noise to the aggregated update for differential privacy.
        """
        noise_stddev = clip_norm * noise_multiplier
        
        noisy_update = {}
        for key, value in aggregated_update.items():
            if torch.is_tensor(value) and value.is_floating_point():
                noise = torch.normal(0, noise_stddev, size=value.shape, device=value.device)
                noisy_update[key] = value + noise
            else:
                noisy_update[key] = value
                
        return noisy_update
        
    def aggregate_with_dp(self, client_updates):
        """
        Aggregate client updates with differential privacy.
        """
        # First, compute model updates (deltas) for each client
        global_state = self.model.state_dict()
        client_deltas = []
        
        for update in client_updates:
            delta = {}
            for key in update['state_dict']:
                if key in global_state:
                    delta[key] = update['state_dict'][key] - global_state[key]
            client_deltas.append({
                'delta': delta,
                'client': update['client']
            })
        
        # Clip each client's update
        clipped_deltas = []
        for delta_info in client_deltas:
            clipped_delta = self.clip_model_update(delta_info['delta'], self.clip_norm)
            clipped_deltas.append({
                'delta': clipped_delta,
                'client': delta_info['client']
            })
        
        # Aggregate clipped deltas
        aggregated_delta = {}
        total_samples = 0
        client_weights = []
        
        for delta_info in clipped_deltas:
            client = delta_info['client']
            _, n_k = client.compute_local_label_distribution()
            client_weights.append(n_k)
            total_samples += n_k
        
        if total_samples > 0:
            normalized_weights = [w / total_samples for w in client_weights]
        else:
            normalized_weights = [1.0 / len(clipped_deltas) for _ in clipped_deltas]
        
        # Initialize aggregated delta
        for key in global_state:
            aggregated_delta[key] = torch.zeros_like(global_state[key])
        
        # Sum weighted deltas
        for i, delta_info in enumerate(clipped_deltas):
            weight = normalized_weights[i]
            for key, value in delta_info['delta'].items():
                if key in aggregated_delta:
                    aggregated_delta[key] += weight * value
        
        # Add noise to aggregated delta
        noisy_delta = self.add_gaussian_noise(aggregated_delta, self.clip_norm, self.noise_multiplier)
        
        # Apply noisy delta to global model
        new_global_state = {}
        for key in global_state:
            if key in noisy_delta:
                new_global_state[key] = global_state[key] + noisy_delta[key]
            else:
                new_global_state[key] = global_state[key]
                
        return new_global_state
        
    def run_round(self, round_idx, participation_rate, local_epochs, lr):
        """
        Run one round of DPFedAvg.
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
        
        # Aggregate with differential privacy
        aggregated_state_dict = self.aggregate_with_dp(client_updates)
        self.model.load_state_dict(aggregated_state_dict)
        
        avg_round_loss = sum(round_losses) / len(round_losses) if round_losses else float('nan')
        return avg_round_loss 