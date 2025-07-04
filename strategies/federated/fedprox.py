from .base import FedAvgStrategy


class FedProxStrategy(FedAvgStrategy):
    def __init__(self, model, clients, device, mu=0.01): # Added mu hyperparameter
        """
        Initialize FedProx strategy.
        
        Args:
            model: Global model to be optimized
            clients: List of clients
            device: Device for computation
            mu: Proximal term coefficient (default 0.01)
        """
        super(FedProxStrategy, self).__init__(model, clients, device)
        self.mu = mu # Store mu
        
    def local_update(self, client, global_model, local_epochs, lr):
        """
        Perform local update with proximal term.
        
        Args:
            client: Client to update
            global_model: Global model state_dict (use state_dict for FedProx loss calculation)
            local_epochs: Number of local epochs
            lr: Learning rate
            
        Returns:
            local_state: Updated model state dict
        """
        # FedProx requires the global model's parameters for the proximal loss term
        return client.local_update(global_model.state_dict(), self.mu, local_epochs, lr) # Pass state_dict and mu 