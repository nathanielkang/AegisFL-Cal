from .base import FedAvgStrategy
from copy import deepcopy
import numpy as np


class MOONStrategy(FedAvgStrategy):
    def __init__(self, model, clients, device, mu=1.0, temperature=0.5): # Added MOON hyperparameters
        """
        Initialize MOON strategy.
        
        Args:
            model: Global model to be optimized
            clients: List of clients
            device: Device for computation
            mu: Contrastive loss weight (default 1.0)
            temperature: Temperature parameter for contrastive loss (default 0.5)
        """
        super(MOONStrategy, self).__init__(model, clients, device)
        self.mu = mu
        self.temperature = temperature
        # MOON requires storing previous global models and local models per client
        self.previous_global_model_state = None
        self.client_previous_local_models = {client.client_id: None for client in clients}

    def run_round(self, round_idx, participation_rate, local_epochs, lr):
        """
        Run one round of MOON.
        Extends FedAvgStrategy.run_round to handle MOON-specific logic if any.
        Currently, it relies on client.local_update_moon which should incorporate MOON logic.
        """
        num_selected_clients = max(1, int(participation_rate * len(self.clients)))
        selected_clients = self.select_clients(num_selected_clients)
        # print(f"MOON Round {round_idx+1}: Selected {len(selected_clients)} clients")

        client_updates = []
        round_losses = [] # List to store losses from clients
        total_samples_for_loss_avg = 0
        weighted_loss_sum = 0

        # self.previous_global_model_state is updated by self.aggregate before this round starts for the *next* one.
        # The client's local_update_moon gets the current global_model object and accesses 
        # self.previous_global_model_state and self.client_previous_local_models from the strategy instance directly or via client.

        for client in selected_clients:
            local_model_state_dict, client_loss = client.local_update_moon(
                deepcopy(self.model), 
                self.previous_global_model_state, 
                self.client_previous_local_models.get(client.client_id, None),
                self.mu, 
                self.temperature, 
                local_epochs, 
                lr
            )
            
            if local_model_state_dict is not None: # Check if update was successful
                client_updates.append({
                    'state_dict': local_model_state_dict,
                    'client': client
                })
                # Update the client's previous local model state for the *next* round
                # This is done after the client has used its *current* previous state for *this* round's update.
                self.client_previous_local_models[client.client_id] = deepcopy(local_model_state_dict) # Store the newly updated local model
            
            if client_loss is not None and not (isinstance(client_loss, float) and np.isnan(client_loss)):
                _, n_k = client.compute_local_label_distribution()
                round_losses.append(client_loss) 
                weighted_loss_sum += client_loss * n_k
                total_samples_for_loss_avg += n_k

        if not client_updates:
            if total_samples_for_loss_avg > 0:
                avg_round_loss = weighted_loss_sum / total_samples_for_loss_avg
            elif round_losses:
                 avg_round_loss = sum(round_losses) / len(round_losses)
            else:
                avg_round_loss = float('nan')
            return avg_round_loss

        # Store the current global model state *before* aggregation for the next round's local updates
        self.previous_global_model_state = deepcopy(self.model.state_dict())
        
        aggregated_state_dict = self.aggregate(client_updates) # Uses FedAvg aggregate
        self.model.load_state_dict(aggregated_state_dict)

        if total_samples_for_loss_avg > 0:
            avg_round_loss = weighted_loss_sum / total_samples_for_loss_avg
        elif round_losses:
            avg_round_loss = sum(round_losses) / len(round_losses)
        else:
            avg_round_loss = float('nan')
        return avg_round_loss

    def local_update(self, client, global_model, local_epochs, lr):
        """
        Perform local update for MOON.
        Requires the previous global model and the client's previous local model.
        
        Args:
            client: Client to update
            global_model: Current global model object
            local_epochs: Number of local epochs
            lr: Learning rate
            
        Returns:
            local_state: Updated model state dict
        """
        prev_global_state = self.previous_global_model_state
        prev_local_state = self.client_previous_local_models.get(client.client_id, None)

        # Perform MOON local update
        # Note: The client's local_update needs to be adapted to accept these additional model states
        # and implement the MOON contrastive loss.
        local_state = client.local_update_moon(
            global_model, 
            prev_global_state, 
            prev_local_state, 
            self.mu, 
            self.temperature, 
            local_epochs, 
            lr
        )
        
        # Update the client's previous local model state for the next round
        # Important: Need to clone or deepcopy to avoid issues with model state changing
        self.client_previous_local_models[client.client_id] = deepcopy(local_state) # Or better: state from model *before* returning

        return local_state

    def aggregate(self, client_updates):
        """
        Standard FedAvg aggregation for MOON, but also update the previous global model state.
        
        Args:
            client_updates: List of dictionaries, each containing 'state_dict' and 'client'.
            
        Returns:
            global_state: Aggregated state dict
        """
        # Store the current global model state *before* aggregation for the next round's local updates
        self.previous_global_model_state = deepcopy(self.model.state_dict()) 

        # Perform standard FedAvg aggregation
        return super().aggregate(client_updates) 