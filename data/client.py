import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
import copy
import numpy as np
import traceback
import scipy.fftpack # Added for DCT/IDCT
from collections import OrderedDict, defaultdict
from typing import List, Dict, Any, Union # Ensure List is imported

# Import HE and crypto primitives - use real crypto if available
try:
    from utils.crypto_real import (
        HEPublicKey, HEPrivateKey, HECiphertext,
        he_encrypt, he_decrypt, he_sum_ciphertexts,
        zkp_generate_statistical_proof, zkp_verify_statistical_proof, ZKPProof
    )
except ImportError as e:
    # NO SIMULATION FALLBACK - FAIL IMMEDIATELY
    raise ImportError(
        f"CRITICAL: Failed to import real crypto module required for client operations.\n"
        f"Error: {e}\n"
        f"Simulation mode is NOT allowed. Please ensure tenseal is installed: pip install tenseal"
    )

# Import the collate function
from utils.data_loader import collate_batch

class Client:
    def __init__(self, client_id, dataset, indices=None, device=None):
        """
        Initialize a client for federated learning.
        
        Args:
            client_id: Unique identifier for the client
            dataset: The dataset to use for training
            indices: The indices of the dataset to use for this client
            device: The device to use for computation
        """
        self.client_id = client_id
        self.device = device if device is not None else torch.device('cpu')
        # print(f"Client {client_id} using device: {self.device}")
        
        try:
            if indices is not None:
                self.dataset = Subset(dataset, indices)
                self.indices = indices
            else:
                self.dataset = dataset
                self.indices = list(range(len(dataset)))
            
            # print(f"Client {client_id} initialized with {len(self.dataset)} samples")
            
            # Create dataloader with smaller batch size for sequence data
            # Determine if it's a text dataset needing the custom collate function
            # Check the base dataset if self.dataset is a Subset
            base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
            
            # Check if it's a text dataset that needs our custom collate function
            # Text datasets will have vocab_size but not be Shakespeare dataset
            # Replace check for ShakespeareDataset with a different approach
            is_text_dataset = hasattr(base_dataset, 'vocab_size') 
            
            # Tentative check: Shakespeare dataset likely has a 'char_to_idx' attribute
            # If this isn't reliable, we can add a simpler flag like is_shakespeare=True to that class
            is_shakespeare = hasattr(base_dataset, 'char_to_idx') if hasattr(base_dataset, 'vocab_size') else False
            
            # Only use collate_batch for text datasets that aren't Shakespeare
            is_text_dataset = is_text_dataset and not is_shakespeare

            batch_size = 16 if is_text_dataset else 32
            collate_fn_to_use = collate_batch if is_text_dataset else None

            if is_text_dataset:
                 # print(f"Client {client_id}: Using custom collate_fn for text data.")
                 pass

            self.dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=(self.device.type == 'cuda'),
                drop_last=True, # Avoid batch size 1 errors with BatchNorm
                collate_fn=collate_fn_to_use # Use custom collate for text
            )
            # print(f"Client {client_id} dataloader created successfully (Batch Size: {batch_size})")
            
        except Exception as e:
            print(f"Error initializing client {client_id}: {str(e)}")
            self.dataset = None
            self.dataloader = None
            self.indices = []
    
    def compute_local_label_distribution(self):
        """
        Compute the distribution of labels in the client's dataset.
        
        Returns:
            label_counts: Dictionary mapping labels to counts
            total_samples: Total number of samples
        """
        try:
            if self.dataset is None:
                return {}, 0
                
            label_counts = {}
            total_samples = 0
            
            # Check if dataset is for Shakespeare based on presence of vocab_size attribute
            is_shakespeare = hasattr(self.dataset.dataset if isinstance(self.dataset, Subset) else self.dataset, 'vocab_size')

            for idx in range(len(self.dataset)):
                # Get data and label - handling both Subset and regular dataset
                if isinstance(self.dataset, Subset):
                    data, label = self.dataset.dataset[self.dataset.indices[idx]]
                else:
                    data, label = self.dataset[idx]
                
                # Handle sequence data (Shakespeare)
                if is_shakespeare:
                    # Ensure label is a tensor and flatten it
                    if isinstance(label, torch.Tensor):
                        target_sequence = label.view(-1) # Flatten the target sequence
                        for char_code in target_sequence:
                            l_item = char_code.item() # Get scalar value of character code
                            label_counts[l_item] = label_counts.get(l_item, 0) + 1
                        # Total samples should represent number of sequences processed, or total characters?
                        # For label distribution, counting characters makes sense.
                        # For n_k in aggregation, we usually need number of sequences/data points.
                        # Let's keep total_samples as number of items processed (characters here).
                        total_samples += target_sequence.numel()
                    else:
                        print(f"Warning: Expected Tensor label for Shakespeare, got {type(label)}")
                else:
                    # Handle regular non-sequence data
                    label_scalar = label
                    if isinstance(label, torch.Tensor):
                        # Ensure it's a scalar tensor before calling .item()
                        if label.numel() == 1:
                            label_scalar = label.item()
                        else:
                            print(f"Warning: Expected scalar label, got tensor with shape {label.shape}")
                            continue # Skip this sample if label is unexpected tensor shape
                            
                    # Convert potential numpy types to standard int/float
                    if hasattr(label_scalar, 'item'): # Handles numpy int/float types
                         label_scalar = label_scalar.item()
                         
                    label_counts[label_scalar] = label_counts.get(label_scalar, 0) + 1
                    total_samples += 1 # Count each data point (sequence or single item)
            
            # Note: For aggregation (like FedAlign n_k), len(self.dataset) might be more appropriate
            # if total_samples was intended to be the number of data points/sequences.
            # We return character counts here for label distribution.
            num_data_points = len(self.dataset)
            return label_counts, num_data_points # Return char counts, but total sequences for n_k
            
        except Exception as e:
            print(f"Error computing label distribution for client {self.client_id}: {str(e)}")
            return {}, 0
    
    def update_model(self, model):
        """
        Update the client's model with a new model.
        
        Args:
            model: The new model
            
        Returns:
            model: A copy of the model with client's parameters
        """
        try:
            self.model = model.to(self.device)
            return deepcopy(model)
        except Exception as e:
            print(f"Error updating model for client {self.client_id}: {str(e)}")
            return model
    
    def local_update(self, model, mu, num_epochs, lr):
        """
        Perform local update on the client's model.
        (Handles FedAvg/FedProx logic based on mu).
        
        Args:
            model: The global model object to update from
            mu: Proximal term coefficient (0.0 for FedAvg)
            num_epochs: Number of local epochs to train for
            lr: Initial learning rate
            
        Returns:
            tuple: (local_model.state_dict(), overall_avg_client_loss)
                     - local_model.state_dict(): The updated model parameters
                     - overall_avg_client_loss: Average training loss over all local epochs
        """
        try:
            original_device = next(model.parameters()).device
            # print(f"Client {self.client_id} local_update - Model originally on device: {original_device}")
            
            local_model = model.to(self.device) # Use the passed model copy
            # device_after_move = next(local_model.parameters()).device
            # print(f"Client {self.client_id} local_update - Model now on device: {device_after_move}")

            # Store global model parameters for FedProx loss
            global_params = None
            if mu > 0: # Only needed for FedProx
                global_params = {name: param.clone().detach() for name, param in local_model.named_parameters()}

            # Use SGD with momentum as decided in previous steps
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            
            # --- Add Learning Rate Scheduler --- 
            # Decay LR by 2% each local epoch
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
            # print(f"Client {self.client_id}: Using StepLR scheduler (step=1, gamma=0.98), Initial LR={lr:.4e}")
            # --- End LR Scheduler --- 
            
            local_model.train()
            total_loss_all_epochs = 0.0
            total_batches_all_epochs = 0

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches_epoch = 0
                for batch_idx, batch_data in enumerate(self.dataloader):
                    optimizer.zero_grad()
                    
                    # Check batch structure
                    is_text_batch = isinstance(batch_data, (list, tuple)) and len(batch_data) == 3
                    
                    if is_text_batch:
                        # AG News: (target, data, offsets)
                        target, data, offsets = batch_data
                        data, target = data.to(self.device), target.to(self.device)
                        offsets = offsets.to(self.device)
                    elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        # Standard: (data, target)
                        data, target = batch_data
                        data, target = data.to(self.device), target.to(self.device)
                        offsets = None # No offsets for standard data
                    else:
                        print(f"Warning (Client {self.client_id}): Skipping unexpected batch format: {type(batch_data)}")
                        continue

                    # Check for empty tensors
                    if data.numel() == 0 or target.numel() == 0:
                        continue

                    # --- Modify model call based on batch type --- 
                    if is_text_batch:
                        output = local_model(data, offsets)
                    else:
                        output = local_model(data)
                    # --- End modification --- 

                    loss = criterion(output, target)

                    # Add FedProx proximal term if mu > 0
                    if mu > 0 and global_params is not None:
                        prox_term = 0.0
                        for name, local_param in local_model.named_parameters():
                            if local_param.requires_grad:
                                global_param = global_params[name]
                                prox_term += torch.sum((local_param - global_param)**2)
                        loss += (mu / 2.0) * prox_term
                    
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item()
                    if not np.isnan(batch_loss) and not np.isinf(batch_loss):
                        epoch_loss += batch_loss
                        num_batches_epoch += 1
                
                total_loss_all_epochs += epoch_loss
                total_batches_all_epochs += num_batches_epoch
                
                # Step the scheduler after each epoch
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                next_lr = scheduler.get_last_lr()[0]
                avg_epoch_loss = epoch_loss / num_batches_epoch if num_batches_epoch > 0 else 0
                # print(f"Client {self.client_id} - Epoch {epoch+1}/{num_epochs} completed. Avg Loss: {avg_epoch_loss:.4f}. LR: {current_lr:.4e} -> {next_lr:.4e}")
                
            overall_avg_client_loss = total_loss_all_epochs / total_batches_all_epochs if total_batches_all_epochs > 0 else 0
            # Return model state dict and the overall average loss
            return local_model.state_dict(), overall_avg_client_loss
        
        except Exception as e:
            print(f"Error during local update for client {self.client_id}: {str(e)}")
            print(traceback.format_exc()) # Print full traceback
            # Return the original model's state_dict if error occurs during training
            # This prevents sending back a potentially corrupted state
            print(f"!!! Client {self.client_id}: Returning original model state due to error during local update.")
            # Return state of the model PASSED IN and a loss of -1 or NaN to indicate error
            return model.state_dict(), float('nan') 

    # --- Placeholder for MOON specific update --- 
    def local_update_moon(self, global_model, prev_global_state, prev_local_state, 
                          mu, temperature, local_epochs, lr):
        # This needs to be implemented based on MOON paper
        # print(f"Client {self.client_id}: MOON local_update not implemented yet.")
        # Fallback to standard update for now
        # Ensure to handle the tuple (state_dict, loss) returned by local_update
        updated_state_dict, loss = self.local_update(global_model, 0.0, local_epochs, lr)
        return updated_state_dict, loss

    def _calculate_model_delta_norm(self, delta_state_dict):
        """Calculates the L2 norm of a model delta (state_dict format)."""
        total_norm_sq = 0.0
        for key in delta_state_dict:
            if delta_state_dict[key].is_floating_point(): # Only consider float tensors for norm
                total_norm_sq += torch.sum(torch.pow(delta_state_dict[key], 2))
        return torch.sqrt(total_norm_sq)

    def local_update_ldpfl(self, global_model_obj, num_epochs, lr, epsilon,
                               layer_centers, layer_radii, T_shuffling_max_delay):
        try:
            local_model = global_model_obj # Use the passed deepcopy directly
            optimizer = optim.SGD(local_model.parameters(), lr=lr)
            criterion = self.get_criterion()
            local_model.train()
            
            total_loss_all_epochs = 0.0
            total_batches_all_epochs = 0
            num_samples_trained = 0

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches_epoch = 0
                for batch_idx, (data, target) in enumerate(self.dataloader):
                    if batch_idx == 0 and epoch == 0: # Count samples only once
                        num_samples_trained += data.size(0)
                        data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0) # General grad clipping
                    optimizer.step()
                    batch_loss = loss.item()
                    if not np.isnan(batch_loss) and not np.isinf(batch_loss):
                        epoch_loss += batch_loss
                        num_batches_epoch += 1
                if num_batches_epoch > 0:
                    total_loss_all_epochs += epoch_loss
                    total_batches_all_epochs += num_batches_epoch
            
            overall_avg_client_loss = total_loss_all_epochs / total_batches_all_epochs if total_batches_all_epochs > 0 else float('nan')

            # LDP-FL Perturbation (Adaptive Clipping + LDP Noise + Parameter Shuffling)
            perturbed_state_dict = OrderedDict()
            final_local_state_dict = local_model.state_dict()

            for name, param_val in final_local_state_dict.items():
                if name in layer_centers and name in layer_radii: # Apply LDP to layers with c,r
                    c_l = layer_centers[name].to(self.device)
                    r_l = layer_radii[name].to(self.device)
                    
                    w_original = param_val.data.to(self.device)
                    original_shape = w_original.shape
                    w_flat = w_original.view(-1)

                    # Explicitly clip all elements first to the [c_l - r_l, c_l + r_l] range
                    # The LDP-FL mechanism definition assumes input w is within this range.
                    w_flat_clipped = torch.clamp(w_flat, c_l - r_l, c_l + r_l)

                    if epsilon <= 1e-6 or r_l.item() < 1e-9: # If epsilon is too small or radius is negligible
                        # Use clipped values, no further LDP perturbation if mechanism is ill-defined
                        w_perturbed_flat = w_flat_clipped
                        # print(f"Client {self.client_id} LDP-FL Layer {name}: Epsilon too small or radius negligible. Using clipped values.")
                    else:
                        exp_epsilon_t = torch.exp(torch.tensor(epsilon, device=w_flat.device))
                        
                        # Check if (exp_epsilon_t - 1) is too close to zero, which makes B_t problematic
                        if (exp_epsilon_t - 1).abs().item() < 1e-9:
                            # This implies epsilon is extremely close to 0.
                            # Fallback to clipped values if B_t would be Inf/NaN.
                            w_perturbed_flat = w_flat_clipped
                            # print(f"Client {self.client_id} LDP-FL Layer {name}: (exp_epsilon - 1) too small. Using clipped values.")
                        else:
                            B_t = (exp_epsilon_t + 1) / (exp_epsilon_t - 1)
                            
                            # Numerator for P_tensor: (w_flat_clipped - c_l) * (exp_epsilon_t - 1) + r_l * (exp_epsilon_t + 1)
                            # Denominator for P_tensor: 2 * r_l * (exp_epsilon_t + 1)
                            # Since r_l.item() >= 1e-9 and (exp_epsilon_t + 1) > 1, denominator should be safe.
                            prob_p_numerator_t = (w_flat_clipped - c_l) * (exp_epsilon_t - 1) + r_l * (exp_epsilon_t + 1)
                            prob_p_denominator_t = 2 * r_l * (exp_epsilon_t + 1)
                            
                            P_tensor = prob_p_numerator_t / prob_p_denominator_t
                            P_tensor = torch.clamp(P_tensor, 0.0, 1.0) # Ensure probabilities are valid [0, 1]

                            random_draws = torch.rand_like(w_flat) # Generate random numbers for Bernoulli trials
                            mask = random_draws < P_tensor # Create a boolean mask

                            # Define the two outcomes of the LDP-FL mechanism
                            outcome1 = c_l + r_l * B_t
                            outcome2 = c_l - r_l * B_t
                            
                            # Apply perturbation using torch.where for vectorized conditional assignment
                            w_perturbed_flat = torch.where(mask, outcome1, outcome2)
                    
                    perturbed_state_dict[name] = w_perturbed_flat.reshape(original_shape)
                else:
                    perturbed_state_dict[name] = param_val.data.clone() # Keep non-LDP layers as is
            
            if T_shuffling_max_delay > 0:
                print(f"Client {self.client_id} LDP-FL: Parameter shuffling with T_max={T_shuffling_max_delay} not yet implemented in this version.")

            # print(f"Client {self.client_id} LDP-FL: Applied LDP perturbation. Epsilon: {epsilon}")
            return perturbed_state_dict, overall_avg_client_loss, num_samples_trained
            
        except Exception as e:
            print(f"Error in Client {self.client_id} local_update_ldpfl: {e}")
            traceback.print_exc()
            return None, float('nan'), 0

    def local_update_smpc(self, global_model_obj, num_epochs, lr):
        # This method is not provided in the original file or the code block
        # It's assumed to exist as it's called in the local_update_ldpfl method
        # Implementation details are not provided in the original file or the code block
        # This method should be implemented based on the original file's logic
        # For now, we'll keep the method signature but return None as the implementation is not provided
        return None, float('nan'), 0

    def get_model_device(self, model):
        return model.state_dict()

    def compute_per_class_statistics_for_calibration(self, feature_dim=10, max_samples_per_class=500):
        """
        Computes class-conditional statistics (mean, covariance, count) for each class in the client's dataset.
        This is the foundation for FL-FCR calibration.
        
        Args:
            feature_dim: Dimension of features to use
            max_samples_per_class: Maximum number of samples to process per class (for efficiency)
            
        Returns:
            Dictionary mapping class labels to dictionaries containing 'mean', 'cov', and 'count'
        """
        class_stats = {}
        
        try:
            # Collect samples by class
            class_samples = {}
            samples_processed = {}
            
            for data, target in self.dataloader:
                # Move to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Handle batch dimension
                batch_size = data.size(0)
                
                # Flatten data if necessary
                current_data_shape = data.shape 
                if len(current_data_shape) > 2:
                    data_flat = data.view(current_data_shape[0], -1) 
                elif len(current_data_shape) == 1: 
                    if current_data_shape[0] > 0: 
                        data_flat = data.unsqueeze(0) # Treat as (1, S)
                    else:
                        data_flat = torch.empty((0,0), device=data.device, dtype=data.dtype) 
                else:  
                        data_flat = data
                
                # Ensure data_flat is 2D and has columns before proceeding
                if data_flat.ndim < 2 or (data_flat.ndim == 2 and data_flat.shape[1] == 0):
                    if data_flat.numel() == 0:
                        # print(f"Client {self.client_id} (project_data): data_flat is empty or has zero columns. Shape: {data_flat.shape}. Skipping batch.")
                        continue
                    else:
                        # This case (e.g., 1D non-empty) should have been reshaped above or indicates an issue.
                        # print(f"Client {self.client_id} (project_data): data_flat has unexpected shape {data_flat.shape}. Skipping batch.")
                        continue
                
                # Process only up to feature_dim features
                if data_flat.shape[1] >= feature_dim:
                    data_processed = data_flat[:, :feature_dim]
                else:
                    # Pad if insufficient features
                    padding = torch.zeros(batch_size, feature_dim - data_flat.shape[1], device=self.device)
                    data_processed = torch.cat((data_flat, padding), dim=1)
                
                # Group samples by class
                for i in range(batch_size):
                    # Get sample and its label
                    sample = data_processed[i].unsqueeze(0)  # Keep batch dimension of 1
                    label = target[i].item()  # Convert to scalar
                    
                    # Initialize class entry if needed
                    if label not in class_samples:
                        class_samples[label] = []
                        samples_processed[label] = 0
                    
                    # Only add sample if we haven't reached max_samples
                    if samples_processed[label] < max_samples_per_class:
                        class_samples[label].append(sample)
                        samples_processed[label] += 1
                    
                    # If we've collected enough samples for all seen classes, stop processing
                    if all(count >= max_samples_per_class for count in samples_processed.values()):
                        break
            
            # Compute statistics for each class
            for label, samples in class_samples.items():
                if len(samples) <= 1:
                    # Skip classes with insufficient samples
                    continue
                
                # Concatenate samples for this class
                class_data = torch.cat(samples, dim=0)
                class_count = class_data.shape[0]
                
                # Compute mean for this class
                class_mean = torch.mean(class_data, dim=0)
                
                # Compute covariance for this class
                # Center the data
                centered_data = class_data - class_mean.unsqueeze(0)
                # Compute covariance matrix: 1/(n-1) * (X-μ)ᵀ(X-μ)
                class_cov = torch.matmul(centered_data.t(), centered_data) / (class_count - 1)
                
                # Store statistics for this class
                class_stats[label] = {
                    'mean': class_mean,
                    'cov': class_cov,
                    'count': class_count
                }
            
            # print(f"Client {self.client_id}: Computed statistics for {len(class_stats)} classes")
            
        except Exception as e:
            print(f"Error computing class statistics for client {self.client_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return class_stats 

    def compute_per_class_raw_stats_for_dphac(self, feature_dim=784, max_samples_per_class=500):
        """
        Computes raw statistics (S_k, O_k, n_k) for each class in the client's dataset.
        
        Args:
            feature_dim: Dimension of features to use (default 784 for MNIST 28x28)
            max_samples_per_class: Maximum number of samples to process per class
            
        Returns:
            Dictionary mapping class labels to dictionaries containing 'S_k', 'O_k', 'n_k'
        """
        class_raw_stats = {}
        
        try:
            # Collect samples by class
            class_samples = {}
            samples_processed = {}
            
            for data, target in self.dataloader:
                # Move to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Handle batch dimension
                batch_size = data.size(0)
                
                # Flatten data if necessary
                current_data_shape = data.shape 
                if len(current_data_shape) > 2:
                    data_flat = data.view(current_data_shape[0], -1) 
                elif len(current_data_shape) == 1: 
                    if current_data_shape[0] > 0: 
                        data_flat = data.unsqueeze(0) # Treat as (1, S)
                    else:
                        data_flat = torch.empty((0,0), device=data.device, dtype=data.dtype) 
                else: 
                        data_flat = data
                
                # Ensure data_flat is 2D and has columns before proceeding
                if data_flat.ndim < 2 or (data_flat.ndim == 2 and data_flat.shape[1] == 0):
                    if data_flat.numel() == 0:
                        # print(f"Client {self.client_id} (project_data): data_flat is empty or has zero columns. Shape: {data_flat.shape}. Skipping batch.")
                        continue
                    else:
                        # This case (e.g., 1D non-empty) should have been reshaped above or indicates an issue.
                        # print(f"Client {self.client_id} (project_data): data_flat has unexpected shape {data_flat.shape}. Skipping batch.")
                        continue
                
                # Process only up to feature_dim features
                if data_flat.shape[1] >= feature_dim:
                    data_processed = data_flat[:, :feature_dim]
                else:
                    # Pad if insufficient features
                    padding = torch.zeros(batch_size, feature_dim - data_flat.shape[1], device=self.device)
                    data_processed = torch.cat((data_flat, padding), dim=1)
                
                # Apply normalization for DP sensitivity bounds
                data_normalized = self._normalize_features_for_dp(data_processed, max_norm=1.0)
                
                # Group samples by class
                for i in range(batch_size):
                    # Get sample and its label
                    sample = data_normalized[i]
                    label = target[i].item()  # Convert to scalar
                    
                    # Initialize class entry if needed
                    if label not in class_samples:
                        class_samples[label] = []
                        samples_processed[label] = 0
                    
                    # Only add sample if we haven't reached max_samples
                    if samples_processed[label] < max_samples_per_class:
                        class_samples[label].append(sample)
                        samples_processed[label] += 1
                    
                    # If we've collected enough samples for all seen classes, stop processing
                    if all(count >= max_samples_per_class for count in samples_processed.values()):
                        break
            
            # Compute raw statistics for each class
            for label, samples in class_samples.items():
                if len(samples) == 0:
                    continue
                
                # Stack samples for this class
                class_data = torch.stack(samples)
                class_count = class_data.shape[0]
                
                # Compute S_k (sum of vectors)
                S_k = torch.sum(class_data, dim=0)
                
                # Compute O_k (sum of outer products)
                O_k = torch.zeros(feature_dim, feature_dim, device=self.device)
                for i in range(class_count):
                    x_i = class_data[i]
                    O_k += torch.outer(x_i, x_i)
                
                # Store raw statistics for this class
                class_raw_stats[label] = {
                    'S_k': S_k,
                    'O_k': O_k,
                    'n_k': torch.tensor(float(class_count), device=self.device)
                }
            
            # print(f"Client {self.client_id}: Computed raw statistics for {len(class_raw_stats)} classes")
            
        except Exception as e:
            print(f"Error computing per-class raw statistics for client {self.client_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return class_raw_stats

    def apply_dp_and_encrypt_per_class_dphac_stats(self, class_raw_stats, dp_params, he_params):
        """
        Applies DP noise and encrypts DPHAC statistics for each class.
        
        Args:
            class_raw_stats: Dictionary mapping class labels to raw statistics
            dp_params: Dictionary with DP parameters (epsilon/delta)
            he_params: Homomorphic encryption parameters
            
        Returns:
            Dictionary mapping class labels to encrypted DP-noised statistics
        """
        class_encrypted_stats = {}
        
        for label, raw_stats in class_raw_stats.items():
            try:
                # Apply DP and encrypt for this class
                encrypted_stat = self.apply_dp_and_encrypt_dphac_stats(raw_stats, dp_params, he_params)
                
                # Store with class label
                class_encrypted_stats[label] = encrypted_stat
                
            except Exception as e:
                print(f"Error applying DP and encrypting statistics for class {label}, client {self.client_id}: {e}")
        
        return class_encrypted_stats 

    def _compute_fisher_information_diag(self, model, criterion):
        """
        (Placeholder) Computes the diagonal of the Fisher Information Matrix (FIM).
        This is a simplified version. A more accurate FIM would require more computation.
        """
        fisher_diag = {name: torch.zeros_like(param.data) for name, param in model.named_parameters() if param.requires_grad}
        model.train() # Ensure model is in train mode for gradient computation

        # Example: Compute empirical Fisher using a few batches from local data
        # This requires gradients of the log-likelihood (often approximated by loss gradients)
        num_fisher_samples = 0
        for _ in range(min(len(self.dataloader), 5)): # Use a few batches
            try:
                batch_data = next(iter(self.dataloader))
                is_text_batch = isinstance(batch_data, (list, tuple)) and len(batch_data) == 3
                
                if is_text_batch:
                    target, data, offsets = batch_data
                    data, target = data.to(self.device), target.to(self.device)
                    offsets = offsets.to(self.device)
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, target = batch_data
                    data, target = data.to(self.device), target.to(self.device)
                    offsets = None
                else:
                    continue
                if data.numel() == 0 or target.numel() == 0:
                    continue

                model.zero_grad()
                if is_text_batch:
                    output = model(data, offsets)
                else:
                    output = model(data)
                
                loss = criterion(output, target)
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        fisher_diag[name] += (param.grad.data ** 2) # Sum of squared gradients
                num_fisher_samples += data.size(0)
            except Exception as e:
                print(f"Client {self.client_id}: Error during Fisher computation batch: {e}")
                continue # Skip batch on error

        if num_fisher_samples > 0:
            for name in fisher_diag:
                fisher_diag[name] /= num_fisher_samples # Average over samples
        
        # print(f"Client {self.client_id}: (Placeholder) Fisher diagonal computed.")
        return fisher_diag

    def local_update_feddpa(self, global_model_obj, param_is_personalized_mask, 
                              num_epochs, lr, dp_C, dp_epsilon, dp_delta, dp_sigma, fisher_threshold):
        """
        Perform local update for FedDPA.
        - Computes Fisher Information.
        - Partitions parameters into personalized (local) and shared (global).
        - Updates personalized parameters locally.
        - Computes delta for shared parameters, applies DP (clipping+noise), returns noisy delta.
        """
        try:
            local_model = deepcopy(global_model_obj).to(self.device)
            initial_shared_params_state_dict = {}
            
            # (Placeholder) Determine/update which parameters are personalized based on Fisher Info
            # This is a simplified local determination; a real FedDPA might get this from server or coordinate
            criterion_for_fisher = nn.CrossEntropyLoss()
            current_fisher_diag = self._compute_fisher_information_diag(local_model, criterion_for_fisher)
            
            # Update param_is_personalized_mask based on current_fisher_diag and fisher_threshold
            # For simplicity, let's assume if a param's avg Fisher val > threshold, it's personalized
            # This is a very simplified interpretation of FedDPA's layer-wise Fisher.
            if param_is_personalized_mask is None: # Initialize if not provided
                param_is_personalized_mask = {}
            for name, fisher_val_tensor in current_fisher_diag.items():
                if torch.mean(fisher_val_tensor) > fisher_threshold:
                    param_is_personalized_mask[name] = True 
                else:
                    param_is_personalized_mask[name] = False
            # print(f"Client {self.client_id} FedDPA - Personalization mask updated. Personalized: {sum(param_is_personalized_mask.values())}/{len(param_is_personalized_mask)}")

            # Store initial state of SHARED parameters (these will be used for delta calculation)
            for name, param in local_model.named_parameters():
                if not param_is_personalized_mask.get(name, False): # If it's a shared parameter
                    initial_shared_params_state_dict[name] = param.data.clone().detach()

            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
            criterion_train = nn.CrossEntropyLoss()
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

            local_model.train()
            total_loss_all_epochs = 0.0
            total_batches_all_epochs = 0

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches_epoch = 0
                for batch_idx, batch_data in enumerate(self.dataloader):
                    optimizer.zero_grad()
                    is_text_batch = isinstance(batch_data, (list, tuple)) and len(batch_data) == 3
                    
                    if is_text_batch:
                        target, data, offsets = batch_data
                        data, target = data.to(self.device), target.to(self.device)
                        offsets = offsets.to(self.device)
                    elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        data, target = batch_data
                        data, target = data.to(self.device), target.to(self.device)
                        offsets = None
                    else: continue
                    if data.numel() == 0 or target.numel() == 0: continue

                    if is_text_batch:
                        output = local_model(data, offsets)
                    else:
                        output = local_model(data)
                    
                    loss = criterion_train(output, target)
                    loss.backward()

                    # FedDPA might have adaptive constraints here - for now, standard optimizer step
                    # Also, personalized params are updated but their gradients are not sent for aggregation.
                    # Shared params' gradients contribute to their update.
                    optimizer.step()
                scheduler.step()

            # Calculate delta for SHARED parameters
            final_local_state_dict = local_model.state_dict()
            shared_param_delta_state_dict = {}
            for name, initial_param_val in initial_shared_params_state_dict.items():
                if final_local_state_dict[name].is_floating_point(): # Ensure it's a float tensor
                    shared_param_delta_state_dict[name] = initial_param_val - final_local_state_dict[name]
            
            # Clip and noise the delta of SHARED parameters
            delta_norm = self._calculate_model_delta_norm(shared_param_delta_state_dict)
            clipping_factor = min(1.0, dp_C / (delta_norm.item() + 1e-6))
            
            noisy_shared_delta_state_dict = {}
            # dp_sigma is the noise multiplier, noise_stddev = dp_sigma * dp_C (clipping bound)
            noise_stddev = dp_sigma * dp_C 

            for name in shared_param_delta_state_dict:
                clipped_delta = shared_param_delta_state_dict[name] * clipping_factor
                noise = torch.normal(0, noise_stddev, size=clipped_delta.shape).to(self.device)
                noisy_shared_delta_state_dict[name] = clipped_delta + noise
            
            # print(f"Client {self.client_id} FedDPA - Update for shared params processed with DP.")
            overall_avg_client_loss = total_loss_all_epochs / total_batches_all_epochs if total_batches_all_epochs > 0 else 0
            return noisy_shared_delta_state_dict, overall_avg_client_loss

        except Exception as e:
            print(f"Error during FedDPA local update for client {self.client_id}: {str(e)}")
            print(traceback.format_exc())
            return None, float('nan')

    def local_update_lap_dp(self, global_model_obj, num_epochs, lr, ldp_epsilon, ldp_sensitivity, ldp_mechanism):
        """
        (Placeholder) Perform local update with Adaptive Localized Differential Privacy.
        The actual LDP mechanism (e.g., on data, gradients, or weights) and its adaptation 
        would depend on the specific FedLAP paper.
        For this placeholder, we simulate adding noise to the model *delta* similar to DP-SGD,
        but using LDP parameters.

        Args:
            global_model_obj: The current global model (PyTorch nn.Module object).
            num_epochs: Number of local epochs to train.
            lr: Learning rate for local training.
            ldp_epsilon: Current LDP epsilon for this client for this round.
            ldp_sensitivity: LDP sensitivity (e.g., clipping norm for model delta if LDP is applied to delta).
            ldp_mechanism: Placeholder for LDP mechanism type (e.g., 'laplace').

        Returns:
            A tuple (model_delta_state_dict, overall_avg_client_loss, num_samples):
                - model_delta_state_dict: The differentially private model delta (state_dict), or None if error.
                - overall_avg_client_loss: Average training loss over local epochs, or float('nan') if error.
                - num_samples: Number of samples in the client's dataset (len(self.dataset)).
        """
        # print(f"Client {self.client_id}: Starting local_update_lap_dp with epsilon={ldp_epsilon:.4f}, sensitivity={ldp_sensitivity}")
        local_model = deepcopy(global_model_obj).to(self.device)
        local_model.train()
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        initial_state_dict = deepcopy(global_model_obj.state_dict())
        total_loss_all_epochs_lapdp = 0.0
        total_batches_all_epochs_lapdp = 0

        for epoch in range(num_epochs):
            epoch_loss_sum_lapdp = 0.0 # Sum of per-batch average losses for the current epoch
            num_batches_epoch_lapdp = 0
            for batch_idx, data_batch in enumerate(self.dataloader):
                try:
                    # --- Data Handling (consistent with other update methods) ---
                    is_text_batch = False
                    if isinstance(data_batch, (list, tuple)):
                        if len(data_batch) == 2: # (data, target)
                            data, target = data_batch
                            offsets = None 
                        elif len(data_batch) == 3: # (target, data, offsets) - for text data like AG_NEWS
                            target, data, offsets = data_batch
                            is_text_batch = True
                        else:
                            # print(f"Client {self.client_id} - local_update_lap_dp: Unexpected data_batch format (length {len(data_batch)}). Skipping batch.")
                            continue
                    else:
                        # print(f"Client {self.client_id} - local_update_lap_dp: Unexpected data_batch type ({type(data_batch)}). Skipping batch.")
                        continue
                    
                    if data.numel() == 0 or target.numel() == 0:
                        # print(f"Client {self.client_id} - local_update_lap_dp: Empty data or target in batch {batch_idx}. Skipping.")
                        continue

                    data, target = data.to(self.device), target.to(self.device)
                    if offsets is not None:
                        offsets = offsets.to(self.device)
                    # --- End Data Handling ---

                    optimizer.zero_grad()
                    if is_text_batch and offsets is not None:
                        output = local_model(data, offsets)
                    else:
                        output = local_model(data)
                    
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=ldp_sensitivity) # Added gradient clipping
                    optimizer.step()
                    batch_loss = loss.item()
                    if not np.isnan(batch_loss) and not np.isinf(batch_loss):
                        epoch_loss_sum_lapdp += batch_loss
                        num_batches_epoch_lapdp += 1
                except Exception as e:
                    print(f"Client {self.client_id} - local_update_lap_dp: Error during training batch {batch_idx} in epoch {epoch}: {e}")
                    traceback.print_exc()
                    # Potentially skip batch or handle error
                    continue 
            
            total_loss_all_epochs_lapdp += epoch_loss_sum_lapdp
            total_batches_all_epochs_lapdp += num_batches_epoch_lapdp
            if num_batches_epoch_lapdp > 0:
                avg_epoch_loss = epoch_loss_sum_lapdp / num_batches_epoch_lapdp
                # print(f"  Client {self.client_id} - LDP Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.4f}")
            else:
                # print(f"  Client {self.client_id} - LDP Epoch {epoch+1}/{num_epochs}, No batches processed.")
                pass # No batches were successfully processed

        # Calculate model delta (updated_model - original_global_model)
        model_delta_state_dict = {}
        current_local_state = local_model.state_dict()
        for key in initial_state_dict:
            model_delta_state_dict[key] = current_local_state[key] - initial_state_dict[key]

        # (Placeholder) Apply LDP: Clipping and Noising the Model Delta
        # 1. Calculate norm of the delta
        delta_norm = self._calculate_model_delta_norm(model_delta_state_dict) # Reuse existing helper
        
        # 2. Clip the delta if its norm exceeds ldp_sensitivity (clipping threshold C for DP-SGD style)
        if delta_norm > ldp_sensitivity and ldp_sensitivity > 0: # ldp_sensitivity acts as C
            clip_factor = ldp_sensitivity / (delta_norm + 1e-9) # Add epsilon for numerical stability
            for key in model_delta_state_dict:
                if model_delta_state_dict[key].is_floating_point(): # Check if tensor is float
                    model_delta_state_dict[key] *= clip_factor
            # print(f"  Client {self.client_id} LDP: Delta clipped from {delta_norm:.4f} to {ldp_sensitivity:.4f}")
        
        # 3. Add noise based on ldp_epsilon and ldp_sensitivity
        if ldp_epsilon > 0: # Only add noise if epsilon is positive
            if ldp_mechanism == 'laplace':
                noise_scale = ldp_sensitivity / ldp_epsilon
                # --- Re-enable noise addition --- 
                for key in model_delta_state_dict:
                    if model_delta_state_dict[key].is_floating_point(): # Check if tensor is float
                        noise = torch.distributions.Laplace(0, noise_scale).sample(model_delta_state_dict[key].shape).to(self.device)
                        model_delta_state_dict[key] += noise
                print(f"  Client {self.client_id} LDP: Added Laplace noise with scale {noise_scale:.4f} (Sensitivity: {ldp_sensitivity}, Epsilon: {ldp_epsilon})")
                # --- End re-enable ---
            # Add other mechanisms (e.g., Gaussian) as elif blocks if needed
            else:
                print(f"  Client {self.client_id} LDP: Unknown LDP mechanism '{ldp_mechanism}'. No noise added.")
        else:
            # print(f"  Client {self.client_id} LDP: Epsilon is zero or negative. No noise added.")
            pass 
            
        # print(f"Client {self.client_id}: local_update_lap_dp finished.")
        overall_avg_client_loss_lapdp = total_loss_all_epochs_lapdp / total_batches_all_epochs_lapdp if total_batches_all_epochs_lapdp > 0 else 0
        return model_delta_state_dict, overall_avg_client_loss_lapdp, len(self.dataset)

    def apply_dp_and_encrypt_dphac_stats(self, raw_stats, dp_params, he_params):
        """
        Applies DP noise and encrypts DPHAC statistics for each class.
        
        Args:
            raw_stats: Dictionary mapping class labels to raw statistics
            dp_params: Dictionary with DP parameters (epsilon/delta)
            he_params: Homomorphic encryption parameters
            
        Returns:
            Dictionary mapping class labels to encrypted DP-noised statistics
        """
        class_encrypted_stats = {}
        
        for label, raw_stat in raw_stats.items():
            try:
                # Apply DP and encrypt for this class
                encrypted_stat = self.apply_dp_and_encrypt_dphac_stats(raw_stat, dp_params, he_params)
                
                # Store with class label
                class_encrypted_stats[label] = encrypted_stat
                
            except Exception as e:
                print(f"Error applying DP and encrypting statistics for class {label}, client {self.client_id}: {e}")
        
        return class_encrypted_stats 

    def local_update_acsfl(self, global_model_obj, num_epochs, lr, epsilon, 
                             layer_centers, layer_radii, eta_compression_ratio):
        """
        Perform local update for ACS-FL strategy.
        Includes local training, adaptive clipping, LDP perturbation (Laplace), 
        and DCT-based compression.

        Args:
            global_model_obj: The global model object (deepcopy) to start training from.
            num_epochs: Number of local epochs.
            lr: Learning rate.
            epsilon: LDP privacy budget.
            layer_centers: Dictionary {layer_name: center_value (c_l)} from server.
            layer_radii: Dictionary {layer_name: radius_value (r_l)} from server.
            eta_compression_ratio: DCT compression ratio (0.0 to 1.0).

        Returns:
            tuple: (processed_state_dict, overall_avg_client_loss, num_samples)
                     - processed_state_dict: Model state_dict after local training, LDP, and DCT.
                     - overall_avg_client_loss: Average training loss.
                     - num_samples: Number of samples in client's dataset.
        """
        try:
            local_model = deepcopy(global_model_obj).to(self.device)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98) # Consistent LR decay

            local_model.train()
            total_loss_all_epochs = 0.0
            total_batches_all_epochs = 0
            gradient_clip_norm = 1.0 # Define a clipping norm for gradients

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches_epoch = 0
                for batch_idx, batch_data in enumerate(self.dataloader):
                    optimizer.zero_grad()
                    is_text_batch = isinstance(batch_data, (list, tuple)) and len(batch_data) == 3
                    
                    if is_text_batch:
                        target, data, offsets = batch_data
                        data, target, offsets = data.to(self.device), target.to(self.device), offsets.to(self.device)
                    elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        data, target = batch_data
                        data, target = data.to(self.device), target.to(self.device)
                        offsets = None
                    else: continue
                    if data.numel() == 0 or target.numel() == 0: continue

                    output = local_model(data, offsets) if is_text_batch else local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=gradient_clip_norm)
                    optimizer.step()
                    batch_loss = loss.item()
                    if not np.isnan(batch_loss) and not np.isinf(batch_loss):
                        epoch_loss += batch_loss
                        num_batches_epoch += 1
                
                total_loss_all_epochs += epoch_loss
                total_batches_all_epochs += num_batches_epoch
                scheduler.step()
                # avg_epoch_loss = epoch_loss / num_batches_epoch if num_batches_epoch > 0 else 0
                # print(f"Client {self.client_id} ACS-FL - Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.4f}")

            overall_avg_client_loss = total_loss_all_epochs / total_batches_all_epochs if total_batches_all_epochs > 0 else 0

            # ACS-FL specific steps: Clipping, LDP Noise, DCT, IDCT (for returning dense model)
            processed_state_dict = local_model.state_dict()
            print_layer_debug = True # Flag to print debug info for only the first processed layer to avoid excessive logs
            with torch.no_grad():
                for name, param in processed_state_dict.items():
                    if name in layer_centers and name in layer_radii: 
                        max_r_l_cap = 5.0 # Max cap for layer radius to control sensitivity
                        if print_layer_debug:
                            print(f"Client {self.client_id} ACS-FL Layer: {name}")
                            print(f"  Original param stats: min={param.data.min():.4f}, max={param.data.max():.4f}, mean={param.data.mean():.4f}, std={param.data.std():.4f}")

                        c_l = layer_centers[name]
                        r_l = layer_radii[name]
                        r_l_effective = torch.clamp(r_l, max=max_r_l_cap) # Cap the radius

                        if print_layer_debug:
                            print(f"  c_l: {c_l:.4f}, r_l (original): {r_l:.4f}, r_l (effective after cap): {r_l_effective:.4f}")
                        
                        # 1. Adaptive Clipping (use r_l_effective for clipping range if desired, or original r_l)
                        # Using original r_l for clipping as per paper, but noise sensitivity based on r_l_effective
                        clipped_param = torch.clamp(param.data, min=c_l - r_l, max=c_l + r_l)
                        if print_layer_debug:
                            print(f"  Clipped param stats (using original r_l): min={clipped_param.min():.4f}, max={clipped_param.max():.4f}, mean={clipped_param.mean():.4f}")
                        
                        # 2. LDP Perturbation (Generalized Piecewise Mechanism - Simplified to Laplace here)
                        if epsilon > 0:
                            sensitivity_effective = 2 * r_l_effective # Sensitivity based on capped radius
                            scale = sensitivity_effective / epsilon
                            if print_layer_debug:
                                print(f"  LDP Noise: sensitivity_eff={sensitivity_effective:.4f}, epsilon={epsilon:.4f}, scale={scale:.4f}")

                            if scale.item() < 1e-9: 
                                noise = torch.zeros_like(clipped_param)
                            else:
                                try:
                                    noise = torch.distributions.Laplace(0, scale).sample(clipped_param.shape).to(self.device)
                                except ValueError as ve:
                                    print(f"Warning (ACS-FL Client {self.client_id}, layer {name}): Laplace noise scale ({scale.item()}) too small or invalid. Adding zero noise. Error: {ve}")
                                    noise = torch.zeros_like(clipped_param)
                            perturbed_param = clipped_param + noise
                        else:
                            perturbed_param = clipped_param 
                            if print_layer_debug:
                                print(f"  Perturbed param stats: min={perturbed_param.min():.4f}, max={perturbed_param.max():.4f}, mean={perturbed_param.mean():.4f}") 

                        # 3. DCT-based Compression & Decompression (for returning dense model)
                        if eta_compression_ratio < 1.0 and perturbed_param.numel() > 1: 
                            original_shape = perturbed_param.shape
                            param_flat_tensor = perturbed_param.view(-1)
                            param_np = param_flat_tensor.cpu().numpy() # Convert to NumPy
                            
                            try:
                                # Use SciPy for DCT
                                dct_coeffs_np = scipy.fftpack.dct(param_np, type=2, norm='ortho')
                                num_coeffs_to_keep = int(param_flat_tensor.numel() * eta_compression_ratio)
                                num_coeffs_to_keep = max(1, num_coeffs_to_keep) # Keep at least one coefficient
                                
                                # Operations on NumPy array
                                indices = np.argsort(np.abs(dct_coeffs_np))[::-1][:num_coeffs_to_keep]
                                dct_coeffs_compressed_np = np.zeros_like(dct_coeffs_np)
                                dct_coeffs_compressed_np[indices] = dct_coeffs_np[indices]
                                
                                # Reconstruct (IDCT)
                                reconstructed_flat_np = scipy.fftpack.idct(dct_coeffs_compressed_np, type=2, norm='ortho')
                                reconstructed_flat_tensor = torch.from_numpy(reconstructed_flat_np).to(self.device).type_as(param.data) 
                                processed_state_dict[name] = reconstructed_flat_tensor.view(original_shape)
                                if print_layer_debug:
                                    print(f"  DCT-IDCT param stats: min={processed_state_dict[name].min():.4f}, max={processed_state_dict[name].max():.4f}, mean={processed_state_dict[name].mean():.4f}")
                            except Exception as e: 
                                print(f"Warning (ACS-FL Client {self.client_id}, layer {name}): DCT/IDCT with SciPy failed: {e}. Using perturbed param without DCT.")
                                processed_state_dict[name] = perturbed_param
                        else:
                            processed_state_dict[name] = perturbed_param 
                        print_layer_debug = False # Print for the first layer only
                    else:
                        processed_state_dict[name] = param.data

            return processed_state_dict, overall_avg_client_loss, len(self.dataset)

        except Exception as e:
            print(f"Error during ACS-FL local update for client {self.client_id}: {str(e)}")
            print(traceback.format_exc())
            return None, float('nan'), 0

    def apply_dp_and_encrypt_dphac_stats(self, raw_stats, dp_params, he_params):
        """
        Applies DP noise and encrypts DPHAC statistics for each class.
        
        Args:
            raw_stats: Dictionary mapping class labels to raw statistics
            dp_params: Dictionary with DP parameters (epsilon/delta)
            he_params: Homomorphic encryption parameters
            
        Returns:
            Dictionary mapping class labels to encrypted DP-noised statistics
        """
        class_encrypted_stats = {}
        
        for label, raw_stat in raw_stats.items():
            try:
                # Apply DP and encrypt for this class
                encrypted_stat = self.apply_dp_and_encrypt_dphac_stats(raw_stat, dp_params, he_params)
                
                # Store with class label
                class_encrypted_stats[label] = encrypted_stat
                
            except Exception as e:
                print(f"Error applying DP and encrypting statistics for class {label}, client {self.client_id}: {e}")
        
        return class_encrypted_stats 

    def local_update_fedmps(self, global_model_obj_theta_t, server_previous_update_direction_us, 
                              num_epochs, lr, epsilon, delta, sigma_gaussian_noise):
        """
        Perform local update for Fed-MPS strategy.
        Includes local training, parameter selection based on update consistency, 
        and Gaussian perturbation on the selected parameters.

        Args:
            global_model_obj_theta_t: The global model *object* at round t (theta_t).
            server_previous_update_direction_us: Server's update direction (u_s = theta_t - theta_{t-1}), or None for round 0.
            num_epochs: Number of local epochs.
            lr: Learning rate.
            epsilon: DP epsilon (for accounting, actual noise from sigma_gaussian_noise).
            delta: DP delta (for accounting).
            sigma_gaussian_noise: Standard deviation for Gaussian noise.

        Returns:
            tuple: (selected_perturbed_delta_ui_tilde, overall_avg_client_loss, num_samples)
                     - selected_perturbed_delta_ui_tilde: The selected and perturbed model delta (state_dict).
                     - overall_avg_client_loss: Average training loss.
                     - num_samples: Number of samples used for training.
        """
        try:
            # Store the state_dict of the passed global model (theta_t) for delta calculation later
            global_model_state_theta_t = deepcopy(global_model_obj_theta_t.state_dict())

            # Use the passed model object directly for local training, after moving to client's device
            local_model = global_model_obj_theta_t.to(self.device)

            optimizer = optim.SGD(local_model.parameters(), lr=lr)
            criterion = self.get_criterion()
            local_model.train()
            
            total_loss_all_epochs = 0.0
            total_batches_all_epochs = 0
            num_samples_trained = 0

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches_epoch = 0
                for batch_idx, (data, target) in enumerate(self.dataloader):
                    if batch_idx == 0 and epoch == 0: # Count samples only once
                        num_samples_trained += data.size(0)
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item()
                    if not np.isnan(batch_loss) and not np.isinf(batch_loss):
                        epoch_loss += batch_loss
                        num_batches_epoch += 1
                if num_batches_epoch > 0:
                    total_loss_all_epochs += epoch_loss
                    total_batches_all_epochs += num_batches_epoch
            
            overall_avg_client_loss = total_loss_all_epochs / total_batches_all_epochs if total_batches_all_epochs > 0 else float('nan')

            # Calculate raw update: u_i = theta_i^{(t+1)} - theta^{(t)}
            final_local_state_theta_i_tplus1 = local_model.state_dict()
            raw_update_ui = OrderedDict()
            for key in global_model_state_theta_t:
                raw_update_ui[key] = final_local_state_theta_i_tplus1[key] - global_model_state_theta_t[key]

            # Parameter Selection via Direction Consistency
            selected_update_ui_hat = OrderedDict()
            
            if server_previous_update_direction_us is None: # Round 0 case
                # print(f"Client {self.client_id} FedMPS: Round 0, selecting all parameters.")
                selected_update_ui_hat = deepcopy(raw_update_ui)
            else:
                # print(f"Client {self.client_id} FedMPS: Selecting parameters based on direction consistency.")
                for key in raw_update_ui:
                    if key in server_previous_update_direction_us:
                        param_raw_update_val = raw_update_ui[key]
                        param_server_update_val = server_previous_update_direction_us[key]
                        
                        # Element-wise sign comparison
                        sign_raw_update = torch.sign(param_raw_update_val)
                        sign_server_update = torch.sign(param_server_update_val)
                        
                        consistency_mask = (sign_raw_update == sign_server_update).type(param_raw_update_val.dtype)
                        
                        selected_update_ui_hat[key] = param_raw_update_val * consistency_mask
                    else:
                        selected_update_ui_hat[key] = torch.zeros_like(raw_update_ui[key])
            
            # Gaussian Perturbation on selected_update_ui_hat
            perturbed_selected_update_ui_tilde = OrderedDict()
            if sigma_gaussian_noise > 0: # Only add noise if sigma is positive (use argument, not self)
                for key in selected_update_ui_hat:
                    param_selected_val = selected_update_ui_hat[key]
                    noise = torch.normal(0, sigma_gaussian_noise, size=param_selected_val.shape).to(self.device)
                    perturbed_selected_update_ui_tilde[key] = param_selected_val + noise
            else:
                perturbed_selected_update_ui_tilde = deepcopy(selected_update_ui_hat) # No noise if sigma is zero

            # print(f"Client {self.client_id} FedMPS: Local update complete.")
            return perturbed_selected_update_ui_tilde, overall_avg_client_loss, num_samples_trained

        except Exception as e:
            print(f"Error in Client {self.client_id} local_update_fedmps: {e}")
            traceback.print_exc()
            return None, float('nan'), 0

    def get_criterion(self):
        # Simple criterion, can be expanded if datasets need different losses
        return nn.CrossEntropyLoss() 

    def get_local_classes(self) -> List[int]:
        """Returns a list of unique class labels present in the client's local dataset."""
        if not self.dataset or len(self.dataset) == 0:
            return []
        labels = []
        for i in range(len(self.dataset)):
            if isinstance(self.dataset, Subset):
                _, label = self.dataset.dataset[self.dataset.indices[i]]
            else:
                _, label = self.dataset[i]
            if isinstance(label, torch.Tensor):
                labels.append(label.item())
            else:
                labels.append(int(label))
        return sorted(list(set(labels)))

    def _extract_features_for_pca_or_projection(self, data_elements, main_model, is_text_model: bool, text_indices=None, offsets=None):
        """Helper to extract features based on model type. `main_model` is the FL model."""
        if is_text_model:
            if text_indices is not None and text_indices.numel() > 0:
                return main_model.embedding(text_indices, offsets).float().detach()
            return None
        else: # Image or Tabular
            model_name = type(main_model).__name__
            if model_name == "MNISTNet":
                x = F.relu(F.max_pool2d(main_model.conv1(data_elements), 2))
                x = F.relu(F.max_pool2d(main_model.conv2(x), 2))
                x = x.view(-1, 1024)
                return F.relu(main_model.fc1(x)).float().detach()
            elif model_name in ["CIFAR10Net", "SVHNNet"]:
                x = F.relu(main_model.conv1(data_elements)); x = main_model.pool(x)
                x = F.relu(main_model.conv2(x)); x = main_model.pool(x)
                x = F.relu(main_model.conv3(x)); x = main_model.pool(x)
                x = x.view(-1, 64 * 4 * 4)
                return F.relu(main_model.fc1(x)).float().detach()
            elif model_name == "CIFAR100Net":
                x = main_model.backbone.conv1(data_elements)
                if hasattr(main_model.backbone, 'bn1'): x = main_model.backbone.bn1(x)
                x = main_model.backbone.relu(x); x = main_model.backbone.maxpool(x)
                x = main_model.backbone.layer1(x); x = main_model.backbone.layer2(x)
                x = main_model.backbone.layer3(x); x = main_model.backbone.layer4(x)
                x = main_model.backbone.avgpool(x)
                return torch.flatten(x, 1).float().detach()
            elif model_name == "CelebANet":
                x = F.relu(main_model.conv1(data_elements)); x = main_model.pool(x)
                x = F.relu(main_model.conv2(x)); x = main_model.pool(x)
                x = F.relu(main_model.conv3(x)); x = main_model.pool(x)
                x = F.relu(main_model.conv4(x)); x = main_model.pool(x)
                x = x.view(-1, 256 * 8 * 8)
                return F.relu(main_model.fc1(x)).float().detach()
            elif data_elements.ndim > 2: # Fallback: Raw flattened pixels for other image types
                return data_elements.view(data_elements.shape[0], -1).float().detach()
            else: # Tabular
                return data_elements.float().detach()

    def prepare_local_statistics_for_smpc_pca(self, model_for_text_embeddings, pca_image_feature_extractor, target_feature_dim_d: int) -> dict:
        """
        Computes local statistics (sum_S, sum_M, n) for each class, using d-dimensional features,
        to be contributed to an SMPC protocol for global PCA. Computations are on self.device.
        For image models, features are extracted from an intermediate layer of model_for_text_embeddings.
        For text models, embeddings from model_for_text_embeddings are used.
        For tabular, raw features are used.
        The pca_image_feature_extractor argument is currently unused by this method's logic but kept for signature consistency.

        Args:
            model_for_text_embeddings: The current global FL model, used for extracting features (text embeddings or image intermediate features).
            pca_image_feature_extractor: (Currently Unused) A potential separate extractor for image PCA features.
            target_feature_dim_d: The target dimension 'd' of features for PCA. Must match the output
                                  dimension of the feature extraction process chosen for the dataset.
        Returns:
            A dictionary: { class_label (int): (sum_S_c_local (Tensor on self.device), 
                                             sum_M_c_local (Tensor on self.device), 
                                             n_c_local (float)), ... }
        """
        stats_by_class = defaultdict(lambda: {
            'sum_s': torch.zeros(target_feature_dim_d, device=self.device, dtype=torch.float32),
            'sum_m': torch.zeros((target_feature_dim_d, target_feature_dim_d), device=self.device, dtype=torch.float32),
            'n': 0.0
        })
        
        # Use model_for_text_embeddings as the primary model for feature extraction
        is_text_model = hasattr(model_for_text_embeddings, 'embedding') and type(model_for_text_embeddings).__name__ == "AGNewsNet"

        if not self.dataloader: return {}
        
        original_model_training_state = False
        if model_for_text_embeddings is not None:
            if next(model_for_text_embeddings.parameters()).device != self.device:
                 model_for_text_embeddings.to(self.device) # Ensure model is on client device
            original_model_training_state = model_for_text_embeddings.training
            model_for_text_embeddings.eval() # Set to eval mode for consistent feature extraction

        for batch_data in self.dataloader:
            target_batch, text_indices, offsets, data_elements = None, None, None, None
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 3 and is_text_model:
                    target_batch, text_indices, offsets = batch_data
                    text_indices, target_batch, offsets = text_indices.to(self.device), target_batch.to(self.device), offsets.to(self.device)
                elif len(batch_data) == 2:
                    data_elements, target_batch = batch_data
                    data_elements, target_batch = data_elements.to(self.device), target_batch.to(self.device)
                else: continue
            else: continue

            # Pass model_for_text_embeddings as the 'main_model' to the helper
            current_batch_features = self._extract_features_for_pca_or_projection(
                data_elements, model_for_text_embeddings, is_text_model, text_indices, offsets
            )

            if current_batch_features is None or current_batch_features.numel() == 0 or target_batch is None: continue
            if not current_batch_features.dtype.is_floating_point: current_batch_features = current_batch_features.float()
            
            if current_batch_features.shape[1] != target_feature_dim_d: 
                continue 

            for i in range(current_batch_features.shape[0]):
                label = target_batch[i].item()
                x_j = current_batch_features[i]
                stats_by_class[label]['sum_s'] += x_j
                stats_by_class[label]['sum_m'] += torch.outer(x_j, x_j)
                stats_by_class[label]['n'] += 1.0
        
        if model_for_text_embeddings is not None: model_for_text_embeddings.train(original_model_training_state)
        output_dict = {label: (data['sum_s'], data['sum_m'], data['n']) for label, data in stats_by_class.items() if data['n'] > 0}
        return output_dict

    def project_data_for_aegisflcal(self, model_for_feature_extraction, U_c_matrix_for_class, d_original_of_U_c, s_c_of_U_c, class_to_project):
        try:
            class_specific_projected_samples = []
            # Use model_for_feature_extraction as the main model for feature extraction logic
            is_text_model = hasattr(model_for_feature_extraction, 'embedding') and type(model_for_feature_extraction).__name__ == "AGNewsNet"

            original_model_training_state_proj = False
            if model_for_feature_extraction is not None: 
                if next(model_for_feature_extraction.parameters()).device != self.device:
                    model_for_feature_extraction.to(self.device)
                original_model_training_state_proj = model_for_feature_extraction.training
                model_for_feature_extraction.eval()

            for batch_data in self.dataloader:
                target_batch, text_indices, offsets, data_elements = None, None, None, None
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) == 3 and is_text_model:
                        target_batch, text_indices, offsets = batch_data
                        text_indices, target_batch, offsets = text_indices.to(self.device), target_batch.to(self.device), offsets.to(self.device)
                    elif len(batch_data) == 2:
                        data_elements, target_batch = batch_data
                        data_elements, target_batch = data_elements.to(self.device), target_batch.to(self.device)
                    else: continue
                else: continue

                features_to_project_in_batch = self._extract_features_for_pca_or_projection(
                    data_elements, model_for_feature_extraction, is_text_model, text_indices, offsets
                )

                if features_to_project_in_batch is None or features_to_project_in_batch.numel() == 0 or target_batch is None: continue
                if not features_to_project_in_batch.dtype.is_floating_point: features_to_project_in_batch = features_to_project_in_batch.float()
                
                if features_to_project_in_batch.shape[1] != d_original_of_U_c: 
                    continue

                mask = (target_batch == class_to_project)
                if not torch.any(mask): continue
                
                features_of_target_class_in_batch = features_to_project_in_batch[mask]
                
                if features_of_target_class_in_batch.numel() > 0:
                    if U_c_matrix_for_class.dtype != features_of_target_class_in_batch.dtype:
                        features_of_target_class_in_batch = features_of_target_class_in_batch.to(U_c_matrix_for_class.dtype)
                    projected_batch_for_class = torch.matmul(features_of_target_class_in_batch, U_c_matrix_for_class)
                    class_specific_projected_samples.append(projected_batch_for_class)
            
            if model_for_feature_extraction is not None: model_for_feature_extraction.train(original_model_training_state_proj)
            
            if class_specific_projected_samples:
                return {class_to_project: torch.cat(class_specific_projected_samples, dim=0)}
            return {}
            
        except Exception as e:
            print(f"Error projecting data for client {self.client_id} (class {class_to_project}): {e}")
            import traceback
            traceback.print_exc()
            if model_for_feature_extraction is not None: model_for_feature_extraction.train(original_model_training_state_proj) 
            return {}

    def compute_projected_stats_encrypt_prove_for_aegisflcal(self, projected_data_dict_all_classes, he_public_key: HEPublicKey, zkp_config, s_c_values_by_class_dict: dict):
        """
        Computes statistics from projected data, encrypts them, and generates ZKPs.
        Uses the new adaptive statistical ZKP without manual parameter tuning.
        """
        client_stats_package = {}
        
        for class_label, class_data_tuple in projected_data_dict_all_classes.items():
            projected_features, U_c_matrix_used = class_data_tuple
            
            if projected_features is None or projected_features.numel() == 0:
                # print(f"Client {self.client_id}, Class {class_label}: No projected features to process.")
                continue

            n_k_c = projected_features.shape[0]
            s_c = projected_features.shape[1] # Dimension of projected features

            # Compute S_k_c_proj (sum of projected features)
            S_k_c_proj = torch.sum(projected_features, dim=0) # Shape: (s_c)

            # Compute O_k_c_proj (sum of outer products of projected features)
            # O_k_c_proj = torch.einsum('ni,nj->ij', projected_features, projected_features) # Shape: (s_c, s_c)
            O_k_c_proj = torch.zeros((s_c, s_c), device=projected_features.device)
            for i in range(n_k_c):
                feature_vec = projected_features[i].unsqueeze(1) # (s_c, 1)
                O_k_c_proj += torch.matmul(feature_vec, feature_vec.T) # (s_c, 1) @ (1, s_c) -> (s_c, s_c)

            # Generate ZKP for statistical properties using adaptive approach
            # No manual parameters needed - the proof will check basic validity
            statistical_zkp = zkp_generate_statistical_proof(
                S_k_c_proj,
                O_k_c_proj,
                n_k_c,
                U_c_matrix_used  # Pass U_c for commitment inside ZKP
            )

            # Encrypt statistics
            c_S_k_c_proj = he_encrypt(S_k_c_proj, he_public_key)
            c_O_k_c_proj = he_encrypt(O_k_c_proj, he_public_key)
            c_n_k_c = he_encrypt(torch.tensor(float(n_k_c)), he_public_key) # Ensure n_k_c is float tensor for HE

            client_stats_package[class_label] = {
                'c_S_k_c_proj': c_S_k_c_proj,
                'c_O_k_c_proj': c_O_k_c_proj,
                'c_n_k_c': c_n_k_c,
                'zkp_proof': statistical_zkp, # Store the new ZKP
                's_c': s_c,
                'U_c_matrix_ref': U_c_matrix_used # For server to know which U_c was used
            }
        return client_stats_package