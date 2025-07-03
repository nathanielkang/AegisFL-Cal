from copy import deepcopy

class Server:
    def __init__(self, model):
        """
        Initialize a server for federated learning.
        
        Args:
            model: The global model to be optimized
        """
        self.model = model
        self.P_y = {}  # Global label distribution
        
    def aggregate(self, client_updates):
        """
        Perform simple FedAvg aggregation.
        
        Args:
            client_updates: List of client model updates (state_dicts)
            
        Returns:
            new_state: Aggregated state dict
        """
        # Create a deep copy of the first client's state
        new_state = deepcopy(client_updates[0])
        
        # Average the parameters
        for key in new_state.keys():
            if 'num_batches_tracked' in key:
                # Skip batch normalization statistics
                continue
                
            # Sum up parameters from all clients
            for i in range(1, len(client_updates)):
                new_state[key] += client_updates[i][key]
                
            # Average
            new_state[key] = new_state[key] / len(client_updates)
            
        return new_state
    
    def update_model(self, new_state):
        """
        Update the server's model state.
        
        Args:
            new_state: New state dict to update with
        """
        self.model.load_state_dict(new_state) 