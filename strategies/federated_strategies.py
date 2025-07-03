import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from copy import deepcopy
from collections import defaultdict, OrderedDict # Added OrderedDict
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR # Add specific schedulers that might be used
from torch.utils.data import Subset, TensorDataset, DataLoader, random_split # For synthetic data handling
from typing import List, Dict, Any, Union # Added for older Python type hint compatibility

# Import real crypto implementation for AegisFL-Cal
try:
    from utils.crypto_real import (
        generate_he_keys, he_encrypt, he_decrypt, he_sum_ciphertexts, 
        HECiphertext, HEPublicKey, HEPrivateKey,
        zkp_generate_projection_proof, zkp_verify_projection_proof, ZKPProof,
        RangeProof,  # Keep RangeProof but remove the non-existent functions
        estimate_smpc_bandwidth, get_crypto_performance_summary, reset_performance_monitor,
        smpc_secure_sum_tensors, smpc_secure_sum_scalars,
        zkp_verify_statistical_proof,
        he_add,
        zkp_generate_statistical_proof, zkp_verify_statistical_proof  # Add these imports
    )
except ImportError as e:
    # NO SIMULATION FALLBACK - FAIL IMMEDIATELY
    raise ImportError(
        f"CRITICAL: Failed to import real crypto module required for AegisFL-Cal.\n"
        f"Error: {e}\n"
        f"Simulation mode is NOT allowed. Please ensure tenseal is installed: pip install tenseal"
    )

# Import other utilities
from models.neural_networks import AGNewsNet
from models.neural_networks import SimpleImagePCALocalFeatureExtractor

# --- Conditional torchtext imports --- 
try:
    import torchtext
    import torchtext.datasets
    torchtext_available = True
    print("torchtext successfully imported")
except ImportError:
    print("torchtext not available - some text-based models and strategies may not work correctly")
    torchtext_available = False
from torch.utils.data import Subset
from collections import Counter
# --- End conditional imports --- 

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



# --- Proposed Privacy-Preserving Strategies ---

class AegisFLCalStrategy(FedAvgStrategy):
    def __init__(self, model, clients, device,
                 subspace_dim=20,                     # This is max_s_c_per_class
                 smpc_pca_config=None,                
                 he_scheme_config=None,               
                 zkp_config=None,                     
                 synthetic_ratio=1.0,                 
                 calibration_epochs=3,              
                 calibration_lr=0.001,              
                 calibration_wd=0.0,                
                 calibration_patience=3,            
                 calibration_lr_scheduler='none',   
                 d_original_hint=None,              # Dimension of features input to PCA (e.g. raw pixel dim, text embed dim)
                 pca_variance_explained_threshold=0.95, # New: Variance threshold for adaptive s_c
                 data_type='tabular'):              # New: Type of data (tabular, image, text)
        """
        Initialize AegisFLCal strategy.
        Args:
            subspace_dim: Max target dimension s_c for per-class subspaces.
            d_original_hint: Original feature dimension d for PCA input features.
            pca_variance_explained_threshold: Min variance to be explained by chosen principal components for adaptive s_c.
            data_type: Type of data ('tabular', 'image', 'text') for synthetic generation method selection.
        """
        super(AegisFLCalStrategy, self).__init__(model, clients, device)
        
        self.max_s_c_per_class = subspace_dim 
        self.synthetic_ratio = synthetic_ratio
        self.d_original_hint = d_original_hint 
        self.pca_variance_explained_threshold = pca_variance_explained_threshold # Store new param
        self.data_type = data_type  # Store data type
        
        # Fallback for d_original_hint if not provided from main.py
        # This inference is basic and might not be ideal for all models, especially CNNs.
        if self.d_original_hint is None:
            try:
                first_param = next(model.parameters())
                if hasattr(model, 'embedding'): # Specific check for AGNewsNet-like models
                     self.d_original_hint = model.embedding.embedding_dim
                elif len(first_param.shape) > 1 and first_param.shape[1] > 1 and first_param.shape[0] > 100: # Heuristic for Linear input features
                    self.d_original_hint = first_param.shape[1]
                else: 
                    # Fallback for typical image models if hint is missing (e.g. MNIST, CIFAR-10)
                    # This is a rough guess; ideally d_original_hint is always set correctly in main.py
                    if hasattr(model, 'conv1'):
                        if model.conv1.in_channels == 1: self.d_original_hint = 784 # MNIST-like
                        elif model.conv1.in_channels == 3: self.d_original_hint = 3072 # CIFAR10-like
                        else: self.d_original_hint = first_param.numel() # Absolute fallback: total elements in first param
                    else: # Non-CNN, non-embedding model without clear input dim from first layer
                         self.d_original_hint = first_param.numel()
                # print(f"Warning: d_original_hint for AegisFLCal was not provided or was None, inferred/defaulted to {self.d_original_hint}. Provide explicitly in main.py for best results.")
            except Exception as e_infer:
                 self.d_original_hint = 784 # Last resort default for safety
                 # print(f"Warning: d_original_hint could not be inferred for AegisFLCal (Error: {e_infer}), defaulted to {self.d_original_hint}.")

        self.pca_feature_extractor = None # Removed custom extractor instantiation

        self.calibration_epochs = calibration_epochs
        self.calibration_lr = calibration_lr
        self.calibration_wd = calibration_wd
        self.calibration_patience = calibration_patience
        self.calibration_lr_scheduler = calibration_lr_scheduler

        self.smpc_pca_config = smpc_pca_config if smpc_pca_config is not None else {}
        self.he_scheme_config = he_scheme_config if he_scheme_config is not None else {}
        self.zkp_config = zkp_config if zkp_config is not None else {}
        
        self.U_subspace_matrices_by_class = {} # Stores U_c for each class_label
        self.s_c_values_by_class = {}      # Stores s_c (actual proj dim) for each class_label
        
        self.he_public_key, self.he_private_key = generate_he_keys()
        self.class_global_stats = {}
        self.synthetic_dataset = None # This will be (list_of_samples_labels_tuples, metadata_dict)
        
        # Add momentum for smoother statistics updates
        self.momentum = 0.9  # Momentum factor for exponential moving average
        self.class_global_stats_ema = {}  # Exponential moving average of stats
        
        # Add option for class-balanced synthetic generation
        self.use_class_balanced_synthetic = True  # Set to True for equal samples per class
        
        print(f"AegisFLCalStrategy: Initialized with data_type={self.data_type}, max_s_c={self.max_s_c_per_class}, d_hint={self.d_original_hint}")
        # print(f"AegisFLCalStrategy: Initialized with max_s_c_per_class={self.max_s_c_per_class}, d_original_hint={self.d_original_hint}, cal_epochs={self.calibration_epochs}, cal_lr={self.calibration_lr}")
        
    def discover_shared_subspace(self):
        """
        Phase 1: Collaborative Per-Class Secure Subspace Discovery (Simulated SMPC-PCA).
        Computes per-class projection matrices U_c based on aggregated local statistics.
        Updates self.U_subspace_matrices_by_class and self.s_c_values_by_class.
        """
        d = self.d_original_hint
        if d is None or d <= 0:
            print(f"CRITICAL ERROR in AegisFLCal discover_shared_subspace: d_original_hint is invalid ({d}). Cannot proceed with PCA.")
            return

        clients_for_pca = self.clients 
        if not clients_for_pca:
            print("Warning (discover_shared_subspace): No clients available for PCA.")
            return

        S_glob_by_class = defaultdict(lambda: torch.zeros(d, device=self.device, dtype=torch.float32))
        M_glob_by_class = defaultdict(lambda: torch.zeros((d, d), device='cpu', dtype=torch.float32))
        n_glob_by_class = defaultdict(lambda: 0.0)
        
        # Also collect global mean for between-class scatter (for discriminative PCA)
        S_glob_total = torch.zeros(d, device=self.device, dtype=torch.float32)
        n_glob_total = 0.0
        
        for client in clients_for_pca:
            # Pass self.pca_feature_extractor (which is now always None) 
            # and self.model (for text embeddings) to client.
            client_local_stats_dict = client.prepare_local_statistics_for_smpc_pca(
                model_for_text_embeddings=self.model, 
                pca_image_feature_extractor=None, # Always None now
                target_feature_dim_d=d
            )
            
            for class_label, (s_local, m_local, n_local) in client_local_stats_dict.items():
                S_glob_by_class[class_label] += s_local.to(self.device) # Ensure s_local is on server's GPU for sum
                M_glob_by_class[class_label] += m_local.cpu()           # Move m_local to CPU and sum on CPU accumulator
                n_glob_by_class[class_label] += n_local
                
                # For global statistics
                S_glob_total += s_local.to(self.device)
                n_glob_total += n_local

        if not n_glob_by_class: 
            print("Warning (discover_shared_subspace): No statistics aggregated from any client for PCA.")
            return

        self.U_subspace_matrices_by_class.clear()
        self.s_c_values_by_class.clear()
        
        # Compute global mean for between-class scatter
        mu_glob_total = S_glob_total / n_glob_total if n_glob_total > 0 else torch.zeros(d, device=self.device)
        
        for class_label in list(n_glob_by_class.keys()): 
            S_final_gpu = S_glob_by_class[class_label]       # This is on self.device (GPU)
            M_final_cpu = M_glob_by_class[class_label]       # This is on CPU
            n_glob_c_val = n_glob_by_class[class_label]

            U_c_for_class_gpu = None
            s_c_for_class_actual = 0

            if n_glob_c_val <= 1 or d == 0: 
                s_c_fallback = min(self.max_s_c_per_class, d) if d > 0 else self.max_s_c_per_class
                if d > 0 and s_c_fallback > 0:
                    random_matrix = torch.randn(d, s_c_fallback, device=self.device)
                    q_fallback, _ = torch.linalg.qr(random_matrix)
                    U_c_for_class_gpu = q_fallback
                    s_c_for_class_actual = s_c_fallback
            else:
                # Move M_final_cpu to GPU for covariance calculation
                M_final_gpu = M_final_cpu.to(self.device)
                mu_glob_c = S_final_gpu / n_glob_c_val # on GPU
                Sigma_glob_c = (M_final_gpu / n_glob_c_val) - torch.outer(mu_glob_c, mu_glob_c) # on GPU
                
                Sigma_glob_c = (Sigma_glob_c + Sigma_glob_c.T) / 2.0
                Sigma_glob_c += torch.eye(d, device=self.device, dtype=Sigma_glob_c.dtype) * 1e-4 # Regularization on self.device

                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_glob_c) # Operates on GPU tensor
                    sorted_eigenvalues, sorted_indices = torch.sort(eigenvalues, descending=True)
                    
                    # Determine s_c based on variance explained
                    explained_variance_ratio = sorted_eigenvalues / torch.sum(sorted_eigenvalues)
                    cumulative_explained_variance = torch.cumsum(explained_variance_ratio, dim=0)
                    
                    s_c_candidate = torch.searchsorted(cumulative_explained_variance, self.pca_variance_explained_threshold).item() + 1
                    # Ensure s_c is at least 1 if possible, and capped by max_s_c and d
                    s_c_actual = min(s_c_candidate, self.max_s_c_per_class, d)
                    s_c_actual = max(1, s_c_actual) # Ensure at least 1 dimension if d > 0
                    if d == 0: s_c_actual = 0
                                        
                    if s_c_actual > 0:
                        selected_indices = sorted_indices[:s_c_actual]
                        U_c_for_class_gpu = eigenvectors[:, selected_indices]
                        s_c_for_class_actual = s_c_actual
                    # else: U_c remains None, s_c_for_class_actual remains 0 if d=0 or s_c_actual calculated as 0

            except Exception as e:
                    print(f"Error during PCA for class {class_label} on {self.device}: {e}. Using random fallback.")
                    s_c_fallback = min(self.max_s_c_per_class, d) if d > 0 else self.max_s_c_per_class
                    if d > 0 and s_c_fallback > 0:
                        random_matrix = torch.randn(d, s_c_fallback, device=self.device)
                        q_fallback, _ = torch.linalg.qr(random_matrix)
                        U_c_for_class_gpu = q_fallback
                        s_c_for_class_actual = s_c_fallback
            
            if U_c_for_class_gpu is not None:
                self.U_subspace_matrices_by_class[class_label] = U_c_for_class_gpu.clone().detach() # Store GPU tensor, detached
                self.s_c_values_by_class[class_label] = s_c_for_class_actual
            else:
                self.U_subspace_matrices_by_class[class_label] = None
                self.s_c_values_by_class[class_label] = 0
        
    def run_round(self, round_idx, participation_rate, local_epochs, lr):
        """
        Run one round of AegisFLCal.
        """
        # print(f"Round {round_idx+1}: Executing AegisFLCal with max_s_c_per_class={self.max_s_c_per_class}")
        
        num_selected_clients = max(1, int(participation_rate * len(self.clients)))
        selected_clients = self.select_clients(num_selected_clients)
        # print(f"Selected {len(selected_clients)} clients for round {round_idx+1}")
        
        if not self.U_subspace_matrices_by_class: # If subspaces not discovered yet (e.g., first round)
            # print("AegisFLCal: No per-class subspaces found, initiating discovery...")
            self.discover_shared_subspace()
            if not self.U_subspace_matrices_by_class:
                print("CRITICAL: Subspace discovery failed to produce any subspaces. Calibration cannot proceed meaningfully.")
                # Depending on desired behavior, could raise error or try to use a global random subspace as ultimate fallback
                # For now, subsequent steps might fail if U_c is None for classes.

        client_updates_payload = [] # For FedAvg model aggregation part
        all_client_he_stats_for_agg = [] # List of dictionaries {class_label: {he_stats...}, ...}
        round_losses = []

        for client in selected_clients:
            local_model_state_dict, client_loss = client.local_update(deepcopy(self.model), 0.0, local_epochs, lr)
            client_updates_payload.append({'state_dict': local_model_state_dict, 'client': client})
            if client_loss is not None and not (isinstance(client_loss, float) and np.isnan(client_loss)):
                round_losses.append(client_loss)

            # --- AegisFL-Cal Specific Part: Projection and HE Statistics --- 
            client_local_classes = client.get_local_classes()
            projected_data_for_client_all_classes = {} # Dict {class: projected_tensor}
            valid_classes_for_projection = []

            for class_c_val in client_local_classes:
                U_c = self.U_subspace_matrices_by_class.get(class_c_val)
                s_c = self.s_c_values_by_class.get(class_c_val)

                if U_c is not None and s_c is not None and s_c > 0:
                    d_original_of_U_c = U_c.shape[0]
                    if d_original_of_U_c != self.d_original_hint:
                        # print(f"Warning: Mismatch for class {class_c_val}. U_c dim0 ({d_original_of_U_c}) != d_original_hint ({self.d_original_hint}). Skipping projection for this class for client {client.client_id}.")
                        continue # Skip if d_original doesn't match, U_c might be from a faulty PCA
                    
                    # Perform projection for this specific class
                    # client.project_data_for_aegisflcal expects model, U_c, d_of_U_c, s_c_of_U_c, class_to_project
                    # U_c is on self.device (GPU). HE ops are simulated and device-agnostic with current crypto_simulation.py.
                    # mu_proj will be on self.device after HE decryption if HE ops preserve device, or if it's moved.
                    # For now, simulated HE ops in crypto_simulation.py return tensors on the device of their inputs.
                    # If inputs to he_decrypt are GPU tensors, output is GPU tensor.
                    one_class_projected_dict = client.project_data_for_aegisflcal(
                        self.model, U_c.to(client.device), d_original_of_U_c, s_c, class_c_val 
                    )
                    # Update the dictionary to include U_c matrix with the projected features
                    for class_label, projected_features in one_class_projected_dict.items():
                        projected_data_for_client_all_classes[class_label] = (projected_features, U_c)
                    valid_classes_for_projection.append(class_c_val)
                # else:
                    # print(f"Client {client.client_id}: No valid U_c or s_c for class {class_c_val}. Skipping projection for this class.")

            if projected_data_for_client_all_classes: # If any class had data projected
                # Client computes HE stats using the s_c values for *its successfully projected classes*
                s_c_config_for_client_stats = {cls: self.s_c_values_by_class.get(cls) for cls in valid_classes_for_projection 
                                               if self.s_c_values_by_class.get(cls) is not None and self.s_c_values_by_class.get(cls) > 0}
                
                if s_c_config_for_client_stats: # Only proceed if there are valid s_c values for the projected classes
                    client_he_statistics = client.compute_projected_stats_encrypt_prove_for_aegisflcal(
                        projected_data_for_client_all_classes,
                        self.he_public_key,
                self.zkp_config, 
                        s_c_config_for_client_stats # Pass the dict of s_c values for relevant classes
                    )
                    if client_he_statistics: # If any stats were actually computed and encrypted
                        all_client_he_stats_for_agg.append(client_he_statistics)
        # 
        # --- Aggregation --- 
        # 1. Aggregate model updates (Standard FedAvg)
        global_model_state_dict = self.model.state_dict() # Default to current if no updates
        if client_updates_payload:
            global_model_state_dict = super().aggregate(client_updates_payload) # Call FedAvgStrategy.aggregate
            self.model.load_state_dict(global_model_state_dict)
        # 
        # 2. Aggregate HE statistics for calibration
        class_calibration_stats_for_round = None
        if all_client_he_stats_for_agg:
            # This aggregate_he_statistics method needs to be adapted from the main aggregate logic
            class_calibration_stats_for_round = self.aggregate_he_statistics(all_client_he_stats_for_agg)

        if class_calibration_stats_for_round:
            # Apply exponential moving average (EMA) to smooth statistics
            if not self.class_global_stats_ema:
                # First round: initialize EMA with current stats
                self.class_global_stats_ema = deepcopy(class_calibration_stats_for_round)
            else:
                # Apply momentum to smooth updates
                for class_label in class_calibration_stats_for_round:
                    if class_label in self.class_global_stats_ema:
                        # EMA update for existing classes
                        old_stats = self.class_global_stats_ema[class_label]
                        new_stats = class_calibration_stats_for_round[class_label]
                        
                        # Update mean with momentum
                        old_stats['mu_proj'] = self.momentum * old_stats['mu_proj'] + (1 - self.momentum) * new_stats['mu_proj']
                        
                        # Update covariance with momentum
                        old_stats['O_proj'] = self.momentum * old_stats['O_proj'] + (1 - self.momentum) * new_stats['O_proj']
                        
                        # Update count (use regular average, not EMA)
                        old_stats['count'] = (old_stats['count'] + new_stats['count']) / 2
                        
                        # Keep the latest subspace and s_c
                        old_stats['subspace'] = new_stats['subspace']
                        old_stats['s_c'] = new_stats['s_c']
                    else:
                        # New class: add to EMA
                        self.class_global_stats_ema[class_label] = deepcopy(class_calibration_stats_for_round[class_label])
            
            # Use EMA stats for synthetic data generation
            self.class_global_stats = self.class_global_stats_ema
            
            # Generate synthetic dataset from the smoothed global stats
            self.synthetic_dataset = self.generate_synthetic_dataset(self.class_global_stats)
            if self.synthetic_dataset and self.synthetic_dataset[0]: # Check if dataset list is not empty
                self.calibrate_model_with_synthetic_data(self.synthetic_dataset)
                # num_classes_calibrated = len(self.class_global_stats)
                # num_synthetic_samples = sum(len(samples) for samples, _ in self.synthetic_dataset[0])
                # print(f"FL-FCR Calibration: Generated {num_synthetic_samples} synthetic samples for {num_classes_calibrated} classes.")
            # else:
                # print("FL-FCR Calibration: No synthetic data generated or dataset was empty.")
        # else:
            # print("FL-FCR Calibration: No class calibration statistics available for this round.")
           
        avg_round_loss = sum(round_losses) / len(round_losses) if round_losses else float('nan')
        
        # Print performance metrics for real crypto
        perf_summary = get_crypto_performance_summary()
        if perf_summary:  # Only print if there are metrics
            print(f"\nAegisFL-Cal Performance Metrics (Round {round_idx+1}):")
            for op_name, metrics in perf_summary.items():
                print(f"  {op_name}:")
                print(f"    Count: {metrics['count']}")
                print(f"    Avg Time: {metrics['avg_time_ms']:.2f} ms")
                print(f"    Total Time: {metrics['total_time_s']:.2f} s")
                if metrics['avg_size_bytes'] > 0:
                    print(f"    Avg Size: {metrics['avg_size_bytes']:.2f} bytes")
                    print(f"    Total Bandwidth: {metrics['total_bandwidth_bytes'] / 1024 / 1024:.2f} MB")
        
        # Estimate SMPC bandwidth for phase 1
        if len(selected_clients) > 0:
            feature_dim = self.d_original_hint
            num_classes = len(self.U_subspace_matrices_by_class)
            if num_classes > 0:
                smpc_bandwidth = estimate_smpc_bandwidth(len(selected_clients), feature_dim, num_classes)
                print(f"  SMPC PCA (Phase 1 - estimated):")
                print(f"    Total Bandwidth: {smpc_bandwidth['total_bandwidth_bytes'] / 1024 / 1024:.2f} MB")
        
        return avg_round_loss

    def aggregate_he_statistics(self, all_client_he_stats_for_agg: List[Dict[Any, Any]]) -> Dict[Any, Any]:
        """
        Aggregates HE statistics from clients, verifying ZKPs first.
        Uses the new adaptive statistical ZKP verification.
        """
        # print(f"AegisFLCalStrategy: Aggregating HE statistics from {len(all_client_he_stats_for_agg)} clients.")
        
        # Phase 1 of adaptive approach: Collect all statistics first (if needed)
        # In a real implementation, this would be done in a separate round
        # For now, we'll verify basic validity only
        
        # Initialize dictionaries for aggregated ciphertexts per class
        aggregated_S_ciphertexts = defaultdict(list)
        aggregated_O_ciphertexts = defaultdict(list)
        aggregated_n_ciphertexts = defaultdict(list)
        # Keep track of U_c matrices and s_c values for valid contributions
        U_c_matrices_per_class = {}
        s_c_values_per_class = {}

        valid_contributions_count = 0
        invalid_zkp_count = 0

        for client_package in all_client_he_stats_for_agg:
            if not isinstance(client_package, dict):
                # print(f"Warning: Expected client_package to be a dict, got {type(client_package)}. Skipping.")
                invalid_zkp_count += 1 # Count as invalid if format is wrong
                continue
            
            # The client package is the stats directly, not wrapped in another dict with 'stats' key
            # client_id_for_log = client_package.get('client_id', 'UnknownClient')
            # print(f"  Processing package from client")

            for class_label, stats_and_proof in client_package.items():
                zkp_proof_obj = stats_and_proof.get('zkp_proof')

                if zkp_proof_obj is None:
                    # print(f"    Client {client_id_for_log}, Class {class_label}: Missing ZKP proof. Skipping.")
                    invalid_zkp_count += 1
                    continue
                    
                # Verify the statistical ZKP proof with adaptive approach
                # For now, we only check basic validity (no adaptive bounds yet)
                is_zkp_valid = zkp_verify_statistical_proof(
                    zkp_proof_obj,
                    adaptive_bounds=None  # In full implementation, would pass bounds from Phase 1
                )

                if is_zkp_valid:
                    # print(f"    Client {client_id_for_log}, Class {class_label}: ZKP VALID. Aggregating stats.")
                    aggregated_S_ciphertexts[class_label].append(stats_and_proof['c_S_k_c_proj'])
                    aggregated_O_ciphertexts[class_label].append(stats_and_proof['c_O_k_c_proj'])
                    aggregated_n_ciphertexts[class_label].append(stats_and_proof['c_n_k_c'])
                    
                    # Store U_c and s_c if this is the first valid contribution for the class
                    if class_label not in U_c_matrices_per_class:
                        U_c_matrices_per_class[class_label] = stats_and_proof.get('U_c_matrix_ref')
                        s_c_values_per_class[class_label] = stats_and_proof.get('s_c')
                    valid_contributions_count += 1
                else:
                    # print(f"    Client {client_id_for_log}, Class {class_label}: ZKP INVALID. Discarding stats.")
                    invalid_zkp_count += 1
        
        # print(f"AegisFLCalStrategy: Total valid ZKP contributions: {valid_contributions_count}, Invalid/Skipped: {invalid_zkp_count}")

        # Perform homomorphic summation for each class
        final_aggregated_stats = {}
        for class_label in aggregated_S_ciphertexts:
            if not aggregated_S_ciphertexts[class_label]: # Should not happen if we have valid contributions
                continue

            summed_S_cipher = he_sum_ciphertexts(aggregated_S_ciphertexts[class_label], self.he_public_key)
            summed_O_cipher = he_sum_ciphertexts(aggregated_O_ciphertexts[class_label], self.he_public_key)
            summed_n_cipher = he_sum_ciphertexts(aggregated_n_ciphertexts[class_label], self.he_public_key)

            if summed_S_cipher and summed_O_cipher and summed_n_cipher:
                final_aggregated_stats[class_label] = {
                    'agg_S_cipher': summed_S_cipher,
                    'agg_O_cipher': summed_O_cipher,
                    'agg_n_cipher': summed_n_cipher,
                    'U_c_matrix': U_c_matrices_per_class.get(class_label), # For back-projection
                    's_c': s_c_values_per_class.get(class_label) # Projected dimension
                }
            else:
                # print(f"Warning: HE summation failed for class {class_label}, possibly due to empty list of ciphertexts after ZKP filtering.")
                pass
        
        # print(f"AegisFLCalStrategy: HE aggregation complete for {len(final_aggregated_stats)} classes.")
        
        # Decrypt the aggregated statistics for use in synthetic data generation
        decrypted_stats = {}
        for class_label, encrypted_stats in final_aggregated_stats.items():
            # Decrypt the aggregated statistics
            agg_S_plain = he_decrypt(encrypted_stats['agg_S_cipher'], self.he_private_key)
            agg_O_plain = he_decrypt(encrypted_stats['agg_O_cipher'], self.he_private_key)
            agg_n_plain = he_decrypt(encrypted_stats['agg_n_cipher'], self.he_private_key)
            
            # Convert n to scalar
            n_total = agg_n_plain.item() if torch.is_tensor(agg_n_plain) else float(agg_n_plain)
            
            if n_total > 0:
                # Compute mean and covariance in projected space
                mu_proj = agg_S_plain / n_total
                Sigma_proj = (agg_O_plain / n_total) - torch.outer(mu_proj, mu_proj)
                
                decrypted_stats[class_label] = {
                    'mu_proj': mu_proj,
                    'O_proj': Sigma_proj,  # This is the covariance matrix
                    'count': n_total,
                    'subspace': encrypted_stats['U_c_matrix'],  # U_c matrix
                    's_c': encrypted_stats['s_c']  # Projected dimension
                }
        
        return decrypted_stats
    
    def generate_synthetic_dataset(self, class_calibration_stats_map: dict):
        """
        Generate synthetic dataset based on the calibrated class statistics (per-class U_c and s_c).
        Uses advanced generation methods based on data type.
        Args:
            class_calibration_stats_map: Dictionary mapping class labels to their stats 
                                        (which includes 'mu_proj', 'O_proj', 'count', 'subspace' (U_c), 's_c').
        Returns:
            Tuple: (synthetic_dataset_list, metadata_dict)
        """
        if not class_calibration_stats_map:
            # print("No calibration statistics available. Cannot generate synthetic dataset.")
            return None, {} # Return empty metadata as well
        
        synthetic_dataset_list = []
        total_synthetic_samples = 0
        
        # Calculate adaptive synthetic ratio based on class distribution skewness
        class_counts = [stats['count'] for stats in class_calibration_stats_map.values()]
        if class_counts:
            # Calculate coefficient of variation as a measure of skewness
            mean_count = np.mean(class_counts)
            std_count = np.std(class_counts)
            cv = std_count / mean_count if mean_count > 0 else 0
            
            # Adaptive ratio: higher CV means more skewed, need more synthetic data
            # CV typically ranges from 0 (uniform) to 2+ (very skewed)
            adaptive_synthetic_ratio = self.synthetic_ratio * (1 + cv)
            adaptive_synthetic_ratio = min(adaptive_synthetic_ratio, 3.0)  # Cap at 3x
            
            print(f"Adaptive synthetic ratio: {adaptive_synthetic_ratio:.2f} (CV={cv:.2f})")
            else:
            adaptive_synthetic_ratio = self.synthetic_ratio
        
        # Calculate total synthetic samples based on mode
        if self.use_class_balanced_synthetic:
            # Class-balanced: equal samples per class
            num_classes = len(class_calibration_stats_map)
            if num_classes > 0:
                # Total synthetic samples = average real samples per class * synthetic ratio * num classes
                avg_samples_per_class = mean_count if class_counts else 100
                samples_per_class = int(avg_samples_per_class * adaptive_synthetic_ratio)
                samples_per_class = max(10, samples_per_class)  # At least 10 per class
                print(f"Class-balanced mode: {samples_per_class} synthetic samples per class")
        
        overall_d_original_for_metadata = self.d_original_hint # Use the hint as the authoritative source for d
        is_image_data_overall = False # Placeholder, logic to determine if it's image data can be complex
        channels_overall, img_size_overall = None, None

        if overall_d_original_for_metadata == 784: # MNIST
            is_image_data_overall = True; channels_overall = 1; img_size_overall = 28
        elif overall_d_original_for_metadata == 3072: # CIFAR/SVHN
            is_image_data_overall = True; channels_overall = 3; img_size_overall = 32

        try:
            from utils.synthetic_generation import generate_synthetic_data_advanced
            print(f"Using advanced synthetic generation method: {self.data_type}")
        except ImportError:
            print("Warning: Could not import advanced synthetic generation. Using Gaussian fallback.")
            generate_synthetic_data_advanced = None
        
        for class_label, stats in class_calibration_stats_map.items():
            mu_proj = stats['mu_proj']      # Shape: (s_c,)
            O_proj = stats['O_proj']        # Shape: (s_c, s_c) - This is Sigma_glob,c,proj
            count = stats['count']          # Scalar
            U_c = stats['subspace']       # Shape: (d, s_c) - Expected on self.device (GPU)
            s_c = stats['s_c']            # Scalar, actual subspace dim for this class

            if U_c is None or s_c == 0 or count == 0:
                # print(f"Skipping synthetic data generation for class {class_label} due to missing U_c, s_c=0, or count=0.")
                continue
            
            if mu_proj.shape[0] != s_c or O_proj.shape[0] != s_c or O_proj.shape[1] != s_c:
                # print(f"Shape mismatch for class {class_label}: mu_proj {mu_proj.shape}, O_proj {O_proj.shape}, expected s_c {s_c}. Skipping.")
                continue

            # Determine synthetic count based on mode
            if self.use_class_balanced_synthetic:
                # Class-balanced: same number for each class
                synthetic_count = samples_per_class
            else:
                # Original mode: proportional to class size with adaptive ratio
                # Use adaptive ratio with class-specific adjustment
                # Minority classes get even more synthetic samples
                class_weight = mean_count / count if count > 0 and mean_count > 0 else 1.0
                class_synthetic_ratio = adaptive_synthetic_ratio * np.sqrt(class_weight)  # Square root to avoid over-amplification
                
                synthetic_count = max(1, int(count * class_synthetic_ratio))
            
            total_synthetic_samples += synthetic_count
            
            try:
                # Use advanced generation method based on data type
                if generate_synthetic_data_advanced is not None:
                    synthetic_samples_proj = generate_synthetic_data_advanced(
                        self.data_type, mu_proj, O_proj, count, synthetic_count, self.device
                    )
                else:
                    # Fallback to original Gaussian generation
                    L = torch.linalg.cholesky(O_proj) # O_proj is on self.device (GPU)
                    z = torch.randn(synthetic_count, s_c, device=self.device) 
                    # mu_proj is on self.device from HE decryption. L.T is from O_proj (self.device)
                    synthetic_samples_proj = mu_proj.unsqueeze(0) + z @ L.T
                
                synthetic_samples_original_space = synthetic_samples_proj @ U_c.T # U_c.T is already on self.device
                
                synthetic_labels = torch.full((synthetic_count,), class_label, dtype=torch.long, device=self.device)
                synthetic_dataset_list.append((synthetic_samples_original_space, synthetic_labels))
                # print(f"Generated {synthetic_count} synthetic samples for class {class_label} (s_c={s_c}, d={U_c.shape[0]}).")
                
            except RuntimeError as e:
                # print(f"Error generating synthetic data for class {class_label} (s_c={s_c}): {e}. Attempting PSD correction.")
                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(O_proj)
                eigenvalues = torch.clamp(eigenvalues, min=1e-5)
                    fixed_O_proj = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
                    
                    # Try advanced generation with fixed covariance
                    if generate_synthetic_data_advanced is not None:
                        synthetic_samples_proj = generate_synthetic_data_advanced(
                            self.data_type, mu_proj, fixed_O_proj, count, synthetic_count, self.device
                        )
                    else:
                        L = torch.linalg.cholesky(fixed_O_proj)
                        z = torch.randn(synthetic_count, s_c, device=self.device)
                synthetic_samples_proj = mu_proj.unsqueeze(0) + z @ L.T
                
                    synthetic_samples_original_space = synthetic_samples_proj @ U_c.T # U_c.T is on self.device
                    synthetic_labels = torch.full((synthetic_count,), class_label, dtype=torch.long, device=self.device)
                    synthetic_dataset_list.append((synthetic_samples_original_space, synthetic_labels))
                except Exception as e_psd:
                    print(f"PSD correction also failed for class {class_label}: {e_psd}. Skipping this class.")
                    pass
        
        # print(f"FL-FCR Synthetic Data Generation complete: {total_synthetic_samples} samples across {len(synthetic_dataset_list)} successfully generated classes.")
        print(f"Synthetic Data Generation ({self.data_type}): {total_synthetic_samples} samples across {len(synthetic_dataset_list)} classes.")
        
        synthetic_dataset_metadata = {
            'is_image_data': is_image_data_overall,
            'channels': channels_overall,
            'img_size': img_size_overall,
            'feature_dim': overall_d_original_for_metadata, # This is d
            'data_type': self.data_type  # Store the data type used
        }
        
        # Return a tuple: (list_of_data_label_pairs, metadata_dict)
        return synthetic_dataset_list, synthetic_dataset_metadata
    
    def calibrate_model_with_synthetic_data(self, synthetic_dataset_tuple):
        """
        Calibrate the global model by training on the synthetic dataset using configured parameters.
        Includes early stopping and learning rate scheduling.
        
        Args:
            synthetic_dataset_tuple: Tuple containing (synthetic_dataset, metadata)
                - synthetic_dataset: List of tuples (samples, labels) for each class
                - metadata: Dictionary with info about data structure
        """
        if not synthetic_dataset_tuple or not synthetic_dataset_tuple[0]:
            # print("No synthetic dataset available. Skipping AegisFLCal calibration.")
            return
            
        synthetic_dataset_list, metadata = synthetic_dataset_tuple
        
        if not synthetic_dataset_list:
            # print("Synthetic dataset list is empty. Skipping AegisFLCal calibration.")
            return

        # Combine all synthetic data for this epoch
        all_samples_list = []
        all_labels_list = []
        
        for samples, labels in synthetic_dataset_list:
            if samples.numel() > 0 and labels.numel() > 0:
                all_samples_list.append(samples)
                all_labels_list.append(labels)
        
        if not all_samples_list or not all_labels_list:
            # print("No valid samples in synthetic_dataset_list. Skipping AegisFLCal calibration.")
            return

        all_samples = torch.cat(all_samples_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long() # Ensure labels are long

        if all_samples.numel() == 0:
            # print("Concatenated synthetic samples are empty. Skipping AegisFLCal calibration.")
            return

        # Create a TensorDataset from the combined synthetic data
        full_synthetic_dataset = TensorDataset(all_samples, all_labels)

        # Split into training and validation sets (e.g., 80% train, 20% validation)
        num_total_synthetic = len(full_synthetic_dataset)
        train_synthetic_dataset, val_synthetic_dataset = None, None # Initialize

        if num_total_synthetic == 0:
            print("No synthetic samples to calibrate on. Skipping calibration.")
            return
        elif num_total_synthetic < 2: # Cannot form a batch of size >= 2 for BatchNorm layers in train mode
            # print(f"Warning: Only {num_total_synthetic} synthetic sample(s). Using all for training, but BatchNorm layers might cause issues or model will be in eval mode.")
            train_synthetic_dataset = full_synthetic_dataset
            # val_synthetic_dataset remains None
        elif num_total_synthetic < 5: # Not enough for a meaningful split, use all for training
            # print(f"Not enough synthetic samples ({num_total_synthetic}) for train/val split. Using all for training.")
            train_synthetic_dataset = full_synthetic_dataset
            # val_synthetic_dataset remains None
        else:
            num_train_synthetic = int(0.8 * num_total_synthetic)
            num_val_synthetic = num_total_synthetic - num_train_synthetic
            if num_train_synthetic < 1 or num_val_synthetic < 1: # Ensure splits are not empty
                # print("Synthetic dataset too small for 80/20 split resulting in non-empty sets. Using all for training.")
                train_synthetic_dataset = full_synthetic_dataset
            else:
            train_synthetic_dataset, val_synthetic_dataset = random_split(
                full_synthetic_dataset, [num_train_synthetic, num_val_synthetic]
            )

        # Prepare training loader
        train_synthetic_loader = None
        can_train_this_round = False
        model_was_switched_to_eval_for_training = False

        if train_synthetic_dataset and len(train_synthetic_dataset) > 0:
            if len(train_synthetic_dataset) < 2:
                # Check if model has BatchNorm layers
                has_batchnorm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in self.model.modules())
                if has_batchnorm:
                    # print(f"Warning: Training synthetic dataset has only {len(train_synthetic_dataset)} sample(s). Model has BatchNorm. Setting model to eval() for this calibration training phase to avoid errors.")
                    # self.model.eval() # Apply to self.model as calibration_model is a deepcopy
                    # model_was_switched_to_eval_for_training = True 
                    # Or, for simplicity, we can skip training if it's problematic
                    # print("Skipping calibration training due to insufficient batch size for BatchNorm layers.")
                    pass # Let it proceed, will use batch_size=1, handled in training loop if needed
                train_batch_size_cal = 1
                train_drop_last_cal = False
                can_train_this_round = True
        else:
                train_batch_size_cal = min(64, len(train_synthetic_dataset) // 10 if len(train_synthetic_dataset) >=10 else len(train_synthetic_dataset))
                train_batch_size_cal = max(2, train_batch_size_cal) # Ensure batch size is at least 2 if possible
                train_drop_last_cal = len(train_synthetic_dataset) >= train_batch_size_cal
                can_train_this_round = True
            
                if can_train_this_round:
                    train_synthetic_loader = DataLoader(train_synthetic_dataset, batch_size=train_batch_size_cal, shuffle=True, drop_last=train_drop_last_cal)
        
        # Prepare validation loader
        val_synthetic_loader = None
        if val_synthetic_dataset and len(val_synthetic_dataset) > 0:
            val_batch_size_cal = min(64, len(val_synthetic_dataset))
            val_batch_size_cal = max(1, val_batch_size_cal) # Val batch can be 1
            val_synthetic_loader = DataLoader(val_synthetic_dataset, batch_size=val_batch_size_cal, shuffle=False)

        if not can_train_this_round or not train_synthetic_loader:
            # print("No trainable synthetic data for calibration this round.")
            # Restore model state if it was switched to eval for training and no training happened
            # if model_was_switched_to_eval_for_training: self.model.train() 
            return

        # Extract metadata for potential reshaping
        is_image_data = metadata.get('is_image_data', False)
        channels = metadata.get('channels', 1)
        img_size = metadata.get('img_size', None)
        feature_dim = metadata.get('feature_dim', None)
        
        calibration_model = deepcopy(self.model)
        calibration_model.to(self.device)
        
        # --- Determine parameters to optimize and forward pass for calibration ---
        params_to_optimize = None
        calibration_forward_pass = None
        model_name_for_cal = type(calibration_model).__name__

        if isinstance(calibration_model, AGNewsNet):
            params_to_optimize = calibration_model.fc.parameters()
            calibration_forward_pass = lambda x: calibration_model.fc(x)
        elif model_name_for_cal == "MNISTNet" and feature_dim == 512:
            params_to_optimize = calibration_model.fc2.parameters()
            calibration_forward_pass = lambda x: calibration_model.fc2(x)
        elif model_name_for_cal in ["CIFAR10Net", "SVHNNet"] and feature_dim == 64:
            params_to_optimize = calibration_model.fc2.parameters()
            calibration_forward_pass = lambda x: calibration_model.fc2(x)
        elif model_name_for_cal == "CIFAR100Net" and feature_dim == 512: # Assuming ResNet-based features
            params_to_optimize = calibration_model.backbone.fc.parameters()
            calibration_forward_pass = lambda x: calibration_model.backbone.fc(x)
        elif model_name_for_cal == "CelebANet" and feature_dim == 512:
            params_to_optimize = calibration_model.fc2.parameters()
            calibration_forward_pass = lambda x: calibration_model.fc2(x)
        else: # Default: calibrate all parameters, assume synthetic data matches full model input
            # This branch is for tabular data or images where PCA was on raw pixels
            # print(f"Calibrating full model for {model_name_for_cal} or feature_dim {feature_dim} not matching specific layer output.")
            params_to_optimize = calibration_model.parameters()
            if is_image_data and img_size is not None and channels is not None and feature_dim is not None:
                expected_raw_image_flat_dim = channels * img_size * img_size
                if feature_dim == expected_raw_image_flat_dim:
                    calibration_forward_pass = lambda x: calibration_model(x.view(x.shape[0], channels, img_size, img_size))
                else:
                    # This case should ideally not be hit if d_original_hint logic is correct for images vs. learned features
                    # print(f"Warning: Image data detected but synthetic feature_dim {feature_dim} does not match raw C*H*W {expected_raw_image_flat_dim}. Using direct pass.")
                    calibration_forward_pass = lambda x: calibration_model(x) # Fallback, might error
            else:
                 calibration_forward_pass = lambda x: calibration_model(x)

        if params_to_optimize is None or calibration_forward_pass is None:
            print(f"Warning: Could not set up calibration for model {model_name_for_cal} with feature_dim {feature_dim}. Skipping calibration.")
            return
        
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.calibration_lr, weight_decay=self.calibration_wd)
        criterion = nn.CrossEntropyLoss()
        
        scheduler = None
        if self.calibration_lr_scheduler == 'plateau' and val_synthetic_loader:
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=max(1, self.calibration_patience // 2), factor=0.1, verbose=True)
        elif self.calibration_lr_scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.calibration_epochs, eta_min=self.calibration_lr*0.01)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(self.calibration_epochs):
            calibration_model.train() # Ensure model is in train mode at start of epoch
            # if model_was_switched_to_eval_for_training: # If we forced eval mode for tiny dataset
            #    calibration_model.eval() 
            
            running_loss = 0.0
            for batch_samples, batch_labels in train_synthetic_loader:
                original_training_state = calibration_model.training
                if batch_samples.shape[0] == 1 and original_training_state:
                    has_batchnorm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in calibration_model.modules())
                    if has_batchnorm:
                        calibration_model.eval()

                batch_samples, batch_labels = batch_samples.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = calibration_forward_pass(batch_samples)
                
                if outputs is None: 
                    if calibration_model.training is False and original_training_state is True: calibration_model.train() 
                    continue # Skip loss computation and backward pass for this batch

                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                if original_training_state is False and calibration_model.training is True: # Should not happen if logic is correct
                     pass #This case means eval was set due to batch_size 1, but optimizer.step might switch it back.
                elif original_training_state is True and calibration_model.training is False: # If BN hack switched it to eval
                    calibration_model.train() # Switch back to train mode after optimizer step if we changed it

                running_loss += loss.item() * batch_samples.size(0)
            
            epoch_loss = running_loss / len(train_synthetic_dataset) if len(train_synthetic_dataset) > 0 else 0
            
            val_loss = -1.0
            if val_synthetic_loader:
                calibration_model.eval() # Crucial: set to eval mode for validation
                current_val_loss = 0.0
                with torch.no_grad():
                    for batch_samples_val, batch_labels_val in val_synthetic_loader:
                        outputs_val = None
                        batch_samples_val, batch_labels_val = batch_samples_val.to(self.device), batch_labels_val.to(self.device)

                        outputs_val = calibration_forward_pass(batch_samples_val)
                        
                        if outputs_val is None: continue

                        loss_val = criterion(outputs_val, batch_labels_val)
                        current_val_loss += loss_val.item() * batch_samples_val.size(0)
                val_loss = current_val_loss / len(val_synthetic_dataset) if len(val_synthetic_dataset) > 0 else 0
                # print(f"AegisFLCal Calibration Epoch {epoch+1}/{self.calibration_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

                if self.calibration_lr_scheduler == 'plateau' and scheduler:
                    scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = deepcopy(calibration_model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if self.calibration_patience > 0 and epochs_no_improve >= self.calibration_patience:
                    # print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss.")
                    break
            else: # No validation loader
                # print(f"AegisFLCal Calibration Epoch {epoch+1}/{self.calibration_epochs}, Train Loss: {epoch_loss:.4f}")
                best_model_state = deepcopy(calibration_model.state_dict()) # Save last state if no validation

            if self.calibration_lr_scheduler == 'cosine' and scheduler:
                scheduler.step()
        
        # After all epochs, if model was globally set to eval for training (tiny dataset case), restore
        # if model_was_switched_to_eval_for_training:
        #    self.model.train() # Restore original global model's training state if it was changed
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            # print("AegisFLCal Model calibration complete! Global model updated with best or last calibration state.")
        else: # Should not happen if training ran at least one epoch
            self.model.load_state_dict(calibration_model.state_dict())
            # print("AegisFLCal Model calibration complete! Global model updated with final calibration state (no best state recorded).")

class LDPFLStrategy(FedAvgStrategy):
    def __init__(self, model, clients, device, ldpfl_epsilon, ldpfl_T_shuffling_max_delay=0):
        super().__init__(model, clients, device)
        self.epsilon = ldpfl_epsilon
        self.T_shuffling_max_delay = ldpfl_T_shuffling_max_delay # Max delay for parameter shuffling
        self.layer_centers = {}
        self.layer_radii = {}
        # Initial computation of layer ranges for the passed model
        self._compute_model_layer_ranges() 
        # print(f"LDPFLStrategy Initialized: epsilon={self.epsilon}, T_shuffling_max_delay={self.T_shuffling_max_delay}")

    def _compute_model_layer_ranges(self):
        """Computes c_l and r_l for each layer of the current global model's weights."""
        self.layer_centers = {}
        self.layer_radii = {}
        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                if param.numel() > 0 and param.is_floating_point(): # Process non-empty float tensors
                    # Filter to likely be weight layers, not biases or batchnorm stats (simple heuristic)
                    if 'weight' in name or 'bias' not in name and param.dim() > 1 :
                        param_data = param.data.view(-1)
                        if param_data.numel() > 0:
                            c_l = torch.mean(param_data)
                            r_l = torch.max(torch.abs(param_data - c_l)) if param_data.numel() > 1 else torch.tensor(0.0, device=param_data.device)
                            # Ensure r_l is not zero to avoid division by zero in LDP mechanism if all values are same
                            if r_l == 0:
                                r_l = torch.tensor(1e-6, device=param_data.device) # A small non-zero radius
                            self.layer_centers[name] = c_l
                            self.layer_radii[name] = r_l
                        else: # Handle scalar params if any, though less common for weights
                            self.layer_centers[name] = param.data.clone()
                            self.layer_radii[name] = torch.tensor(1e-6, device=param.data.device)
        # print(f"LDPFL Debug: Computed layer ranges. Centers: {self.layer_centers}, Radii: {self.layer_radii}")


    def run_round(self, round_idx, participation_rate, local_epochs, lr):
        num_selected_clients = max(1, int(participation_rate * len(self.clients)))
        selected_clients = self.select_clients(num_selected_clients)
        
        client_updates_info = [] # To store (state_dict_processed, client_loss, num_client_samples)
        round_total_loss = 0.0
        round_total_samples = 0

        # Compute c,r for the current global model before sending to clients
        self._compute_model_layer_ranges()

        for client in selected_clients:
            # Client performs local update with LDP.
            # It should return (processed_state_dict, avg_training_loss, num_samples).
            processed_state_dict, client_loss, num_samples = client.local_update_ldpfl(
                global_model_obj=deepcopy(self.model), 
                num_epochs=local_epochs, 
                lr=lr, 
                epsilon=self.epsilon,
                layer_centers=self.layer_centers, # Pass computed centers
                layer_radii=self.layer_radii,     # Pass computed radii
                T_shuffling_max_delay=self.T_shuffling_max_delay
            )
            if processed_state_dict is not None:
                client_updates_info.append({
                    'local_delta_or_model': processed_state_dict, # LDPFL likely returns full model, not delta
                    'client': client,
                    'num_samples': num_samples 
                })
            if client_loss is not None and not (isinstance(client_loss, float) and np.isnan(client_loss)) and num_samples > 0:
                round_total_loss += client_loss * num_samples
                round_total_samples += num_samples
        
        if not client_updates_info:
            # print("LDPFLStrategy: No client updates received or all failed.")
            # Return current model state and NaN loss if no updates
            current_model_state_dict = self.model.state_dict()
            return current_model_state_dict, float('nan')

        # Aggregate processed updates (these are likely full models, not deltas)
        # LDP-FL paper implies server aggregates perturbed weights (i.e. models)
        aggregated_model_state = self.aggregate_models([info['local_delta_or_model'] for info in client_updates_info],
                                                       [info['num_samples'] for info in client_updates_info])
        
        # Update global model
        self.model.load_state_dict(aggregated_model_state)
        
        avg_train_loss = round_total_loss / round_total_samples if round_total_samples > 0 else float('nan')
        # Unlike other strategies that return delta, this one applies the aggregated model directly.
        # For consistency, we might need to consider if it should return a delta or if the framework handles full model updates.
        # For now, assuming the framework expects the strategy to update self.model and return the aggregated state and loss.
        return aggregated_model_state, avg_train_loss

    def aggregate_models(self, model_states, client_sample_sizes):
        if not model_states:
            return self.model.state_dict() # Return current model if no updates

        total_samples = sum(client_sample_sizes)
        if total_samples == 0:
            return self.model.state_dict()

        aggregated_state_dict = OrderedDict()
        for key in model_states[0].keys():
            weighted_sum = torch.zeros_like(model_states[0][key], dtype=torch.float32)
            for i, state_dict in enumerate(model_states):
                weighted_sum += state_dict[key] * (client_sample_sizes[i] / total_samples)
            aggregated_state_dict[key] = weighted_sum
        return aggregated_state_dict


class SMPCFedAvgStrategy(FedAvgStrategy):
    def __init__(self, model, clients, device, smpc_simulation_config=None):
        super().__init__(model, clients, device)
        self.smpc_simulation_config = smpc_simulation_config if smpc_simulation_config is not None else {}
        print(f"SMPCFedAvgStrategy: Initialized (SMPC simulation mode). Config: {self.smpc_simulation_config}")

    def aggregate(self, client_updates):
        print(f"SMPCFedAvgStrategy: Simulating Secure Multi-Party Computation for aggregating {len(client_updates)} updates.")
        aggregated_state_dict = super().aggregate(client_updates)
        print("SMPCFedAvgStrategy: SMPC aggregation simulation complete.")
        return aggregated_state_dict

class ACSFLStrategy(FedAvgStrategy):
    def __init__(self, model, clients, device, epsilon=1.0, compression_ratio_eta=0.1, num_clusters_m=1):
        super().__init__(model, clients, device)
        self.epsilon = epsilon 
        self.eta = compression_ratio_eta 
        self.m = num_clusters_m 
        self.layer_centers = {}
        self.layer_radii = {}
        self._compute_model_layer_ranges() 
        print(f"ACSFLStrategy: Initialized with epsilon={self.epsilon}, eta={self.eta}, num_clusters={self.m}")

    def _compute_model_layer_ranges(self):
        """Computes c_l and r_l for each layer of the current global model (ACSFL version)."""
        self.layer_centers = {}
        self.layer_radii = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters(): 
                if param.requires_grad and param.dim() > 1: 
                    flat_param = param.data.view(-1)
                    if flat_param.numel() > 0:
                        self.layer_centers[name] = torch.mean(flat_param)
                        current_radius = torch.max(torch.abs(flat_param - self.layer_centers[name]))
                        if current_radius.item() < 1e-9: 
                            self.layer_radii[name] = torch.tensor(1e-6, device=param.device) 
            else:
                            self.layer_radii[name] = current_radius

    def run_round(self, round_idx, participation_rate, local_epochs, lr):
        num_selected_clients = max(1, int(participation_rate * len(self.clients)))
        selected_clients = self.select_clients(num_selected_clients)
        self._compute_model_layer_ranges()
        client_processed_updates_for_aggregation = [] 
        round_total_loss = 0.0
        round_total_samples = 0
        for client in selected_clients:
            processed_state_dict, client_avg_loss, client_num_samples = client.local_update_acsfl(
                global_model_obj=deepcopy(self.model),
                num_epochs=local_epochs, 
                lr=lr,
                epsilon=self.epsilon,
                layer_centers=self.layer_centers,
                layer_radii=self.layer_radii,
                eta_compression_ratio=self.eta 
            )
            if processed_state_dict is not None:
                client_processed_updates_for_aggregation.append({
                    'state_dict': processed_state_dict, 
                    'client': client,
                    'num_samples': client_num_samples 
                })
            if client_avg_loss is not None and not (isinstance(client_avg_loss, float) and np.isnan(client_avg_loss)) and client_num_samples > 0:
                round_total_loss += client_avg_loss * client_num_samples
                round_total_samples += client_num_samples
        if not client_processed_updates_for_aggregation:
            avg_round_loss = round_total_loss / round_total_samples if round_total_samples > 0 else float('nan')
            return self.model.state_dict(), avg_round_loss 
        aggregated_model_state = self._aggregate_acsfl_models(client_processed_updates_for_aggregation)
        self.model.load_state_dict(aggregated_model_state)
        avg_round_loss = round_total_loss / round_total_samples if round_total_samples > 0 else float('nan')
        return aggregated_model_state, avg_round_loss

    def _aggregate_acsfl_models(self, client_updates_info):
        model_states = [info['state_dict'] for info in client_updates_info]
        client_sample_sizes = [info['num_samples'] for info in client_updates_info]
        if not model_states:
            return self.model.state_dict()
        total_samples = sum(client_sample_sizes)
        if total_samples == 0:
            return self.model.state_dict()
        aggregated_state_dict = OrderedDict()
        for key in model_states[0].keys():
            accumulator = torch.zeros_like(model_states[0][key].float(), device=model_states[0][key].device)
            for i, state_dict in enumerate(model_states):
                param_to_add = state_dict[key].to(accumulator.device).float()
                accumulator += param_to_add * (client_sample_sizes[i] / total_samples)
            aggregated_state_dict[key] = accumulator.type_as(model_states[0][key])
        return aggregated_state_dict

class FedMPSStrategy(FedAvgStrategy):
    def __init__(self, model, clients, device, 
                 fedmps_epsilon, fedmps_delta, fedmps_sigma_gaussian_noise):
        super().__init__(model, clients, device)
        self.epsilon = fedmps_epsilon
        self.delta = fedmps_delta
        self.sigma_gaussian_noise = fedmps_sigma_gaussian_noise
        self.server_previous_update_direction = None 
        self.global_model_state_at_round_start = None 
        print(f"FedMPSStrategy Initialized: epsilon={self.epsilon}, delta={self.delta}, sigma_gaussian_noise={self.sigma_gaussian_noise}")

    def aggregate_selected_deltas(self, client_selected_deltas_info):
        """ 
        Aggregates selected and perturbed deltas from clients using a simple average.
        Assumes client_selected_deltas_info is a list of dictionaries,
        each containing 'selected_perturbed_delta' and 'client'.
        """
        if not client_selected_deltas_info:
            return {key: torch.zeros_like(param) for key, param in self.model.state_dict().items() if param.is_floating_point()}
        sample_delta = None
        for info in client_selected_deltas_info:
            if info.get('selected_perturbed_delta') is not None:
                sample_delta = info['selected_perturbed_delta']
                break
        if sample_delta is None: 
             return {key: torch.zeros_like(param) for key, param in self.model.state_dict().items() if param.is_floating_point()}
        aggregated_delta = {
            key: torch.zeros_like(delta_param.float(), device=delta_param.device) 
            for key, delta_param in sample_delta.items() 
            if delta_param.is_floating_point()
        }
        num_valid_updates = 0
        for info in client_selected_deltas_info:
            delta = info.get('selected_perturbed_delta')
            if delta is not None:
                num_valid_updates += 1
                for key in aggregated_delta: 
                    if key in delta:
                        delta_param_val = delta[key].to(aggregated_delta[key].device).float()
                        aggregated_delta[key] += delta_param_val
            if num_valid_updates > 0:
                for key in aggregated_delta:
                    aggregated_delta[key] /= num_valid_updates
                    aggregated_delta[key] = aggregated_delta[key].type_as(sample_delta[key])
        return aggregated_delta

    def run_round(self, round_idx, participation_rate, local_epochs, lr):
        num_selected_clients = max(1, int(participation_rate * len(self.clients)))
        selected_clients = self.select_clients(num_selected_clients)
        theta_t_state_dict = deepcopy(self.model.state_dict()) 
        client_outputs = [] 
        round_total_loss = 0.0
        round_total_samples = 0
        server_update_direction_for_clients = self.server_previous_update_direction
        if round_idx == 0:
            # print("FedMPSStrategy: Round 0, server_previous_update_direction is None. Clients will select all parameters.")
            pass # Added pass
        for client in selected_clients:
            selected_perturbed_delta, client_loss, num_samples = client.local_update_fedmps(
                global_model_obj_theta_t=deepcopy(self.model), 
                server_previous_update_direction_us=server_update_direction_for_clients,
                num_epochs=local_epochs, 
                lr=lr, 
                epsilon=self.epsilon, 
                delta=self.delta, 
                sigma_gaussian_noise=self.sigma_gaussian_noise
            )
            if selected_perturbed_delta is not None:
                client_outputs.append({
                    'selected_perturbed_delta': selected_perturbed_delta,
                    'client': client, 
                    'num_samples': num_samples 
                })
            if client_loss is not None and not (isinstance(client_loss, float) and np.isnan(client_loss)) and num_samples > 0:
                round_total_loss += client_loss * num_samples
                round_total_samples += num_samples
        if not client_outputs:
            # print("FedMPSStrategy: No client updates received or all failed.")
            self.server_previous_update_direction = OrderedDict() 
            for key, param in self.model.state_dict().items():
                 self.server_previous_update_direction[key] = torch.zeros_like(param)
            return self.model.state_dict(), float('nan')
        aggregated_selected_delta = self.aggregate_selected_deltas(client_outputs)
        new_global_state_theta_t_plus_1 = OrderedDict()
        for key in theta_t_state_dict:
            if key in aggregated_selected_delta:
                delta_val = aggregated_selected_delta[key].to(theta_t_state_dict[key].device)
                new_global_state_theta_t_plus_1[key] = theta_t_state_dict[key] + delta_val
                    else:
                new_global_state_theta_t_plus_1[key] = theta_t_state_dict[key]
        self.model.load_state_dict(new_global_state_theta_t_plus_1)
        self.server_previous_update_direction = OrderedDict()
        current_model_state_theta_t_plus_1 = self.model.state_dict()
        for key in current_model_state_theta_t_plus_1:
            self.server_previous_update_direction[key] = current_model_state_theta_t_plus_1[key] - theta_t_state_dict[key]
        avg_train_loss = round_total_loss / round_total_samples if round_total_samples > 0 else float('nan')
        return self.model.state_dict(), avg_train_loss

class DPFedAvgStrategy(FedAvgStrategy):
    """Differentially Private Federated Averaging with Gaussian Mechanism."""
    
    def __init__(self, model, clients, device, epsilon=1.0, delta=1e-5, clip_norm=1.0, noise_multiplier=None):
        """
        Initialize DP-FedAvg strategy.

        Args:
            model: Global model to be optimized
            clients: List of clients
            device: Device for computation
            epsilon: Privacy budget (smaller = more private)
            delta: Privacy parameter for (epsilon, delta)-DP
            clip_norm: Clipping norm for gradients/updates
            noise_multiplier: Gaussian noise multiplier (if None, computed from epsilon)
        """
        super().__init__(model, clients, device)
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        
        # Compute noise multiplier from privacy budget if not provided
        if noise_multiplier is None:
            # Using the formula from "The Algorithmic Foundations of Differential Privacy"
            # For Gaussian mechanism: sigma = clip_norm * sqrt(2 * log(1.25/delta)) / epsilon
            self.noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        else:
            self.noise_multiplier = noise_multiplier
            
        print(f"DP-FedAvg initialized: epsilon={epsilon}, delta={delta}, clip_norm={clip_norm}, noise_multiplier={self.noise_multiplier:.4f}")
    
    def clip_model_update(self, update_dict, clip_norm):
        """
        Clip the norm of model update to ensure bounded sensitivity.
        
        Args:
            update_dict: Dictionary of parameter updates
            clip_norm: Maximum allowed L2 norm
            
        Returns:
            Clipped update dictionary and the scaling factor used
        """
        # Compute total norm of the update
        total_norm = 0.0
        for key, param in update_dict.items():
            if param.is_floating_point():
                total_norm += torch.sum(param ** 2).item()
        total_norm = np.sqrt(total_norm)
        
        # Compute clipping factor
        clip_factor = min(1.0, clip_norm / (total_norm + 1e-6))
        
        # Apply clipping
        clipped_update = {}
        for key, param in update_dict.items():
            clipped_update[key] = param * clip_factor
            
        return clipped_update, clip_factor
    
    def add_gaussian_noise(self, aggregated_update, clip_norm, noise_multiplier):
        """
        Add Gaussian noise to the aggregated update for differential privacy.

        Args:
            aggregated_update: Aggregated model update
            clip_norm: Clipping norm used
            noise_multiplier: Noise multiplier for Gaussian mechanism
            
        Returns:
            Noisy update
        """
        noise_scale = clip_norm * noise_multiplier
        noisy_update = {}
        
        for key, param in aggregated_update.items():
            if param.is_floating_point():
                # Add Gaussian noise
                noise = torch.normal(0, noise_scale, size=param.shape, device=param.device)
                noisy_update[key] = param + noise
            else:
                # Non-floating point parameters (e.g., batch norm stats) are not perturbed
                noisy_update[key] = param
                
        return noisy_update
    
    def aggregate_with_dp(self, client_updates):
        """
        Aggregate client updates with differential privacy.
        
        Args:
            client_updates: List of dictionaries containing 'state_dict' and 'client'
            
        Returns:
            DP-aggregated global state
        """
        if not client_updates:
            return self.model.state_dict()
        
        # Get current global model state
        global_state = self.model.state_dict()
        
        # Compute updates (deltas) from each client
        client_deltas = []
        client_weights = []
        total_samples = 0
        
        for update in client_updates:
            client = update['client']
            client_state = update['state_dict']
            
            # Compute delta: client_model - global_model
            delta = {}
            for key in global_state.keys():
                if key in client_state:
                    delta[key] = client_state[key] - global_state[key]
                    
            # Clip the delta
            clipped_delta, clip_factor = self.clip_model_update(delta, self.clip_norm)
            client_deltas.append(clipped_delta)
            
            # Get client weight (number of samples)
            _, n_k = client.compute_local_label_distribution()
            client_weights.append(n_k)
            total_samples += n_k
        
        # Compute weighted average of clipped deltas
        aggregated_delta = {}
        for key in global_state.keys():
            aggregated_delta[key] = torch.zeros_like(global_state[key])
            
        for i, delta in enumerate(client_deltas):
            weight = client_weights[i] / total_samples if total_samples > 0 else 1.0 / len(client_deltas)
            for key in delta.keys():
                if key in aggregated_delta:
                    aggregated_delta[key] += weight * delta[key]
        
        # Add Gaussian noise for differential privacy
        noisy_delta = self.add_gaussian_noise(aggregated_delta, self.clip_norm, self.noise_multiplier)
        
        # Apply noisy delta to global model
        new_global_state = {}
        for key in global_state.keys():
            if key in noisy_delta:
                new_global_state[key] = global_state[key] + noisy_delta[key]
                else:
                new_global_state[key] = global_state[key]
                
        return new_global_state
    
    def run_round(self, round_idx, participation_rate, local_epochs, lr):
        """
        Run one round of DP-FedAvg.

        Args:
            round_idx: Current round index
            participation_rate: Fraction of clients to select
            local_epochs: Number of local epochs
            lr: Learning rate
            
        Returns:
            Average round loss
        """
        # Select clients
        num_selected_clients = max(1, int(participation_rate * len(self.clients)))
        selected_clients = self.select_clients(num_selected_clients)

        client_updates = []
        round_losses = []

        # Perform local updates
        for client in selected_clients:
            local_model_state_dict, client_loss = client.local_update(
                deepcopy(self.model), 0.0, local_epochs, lr
            )
            
            client_updates.append({
                'state_dict': local_model_state_dict,
                'client': client
            })
            
            if client_loss is not None and not np.isnan(client_loss):
                round_losses.append(client_loss)
        
        if not client_updates:
            return float('nan')
        
        # Aggregate with DP
        dp_aggregated_state = self.aggregate_with_dp(client_updates)
        
        # Update global model
        self.model.load_state_dict(dp_aggregated_state)
        
        # Return average loss
        avg_round_loss = sum(round_losses) / len(round_losses) if round_losses else float('nan')
        return avg_round_loss