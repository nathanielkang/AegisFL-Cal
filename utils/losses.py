import torch
import torch.nn as nn
import torch.nn.functional as F

class FedAlignLoss(nn.Module):
    """
    FedAlign custom loss function that reweights the loss based on class frequencies
    to handle class imbalance in local client data.
    
    The class-balanced loss is defined as:
    L_cb = sum_i (1 - beta) / (1 - beta^n_i) * L_i
    
    where:
    - L_i is the original loss for class i
    - n_i is the number of samples for class i
    - beta is a hyperparameter (not used in this implementation)
    
    This implementation uses a simpler formula:
    L_cb = sum_i (1 / (n_i + epsilon)) * L_i
    
    where epsilon is a small constant for numerical stability.
    """
    def __init__(self, label_counts, num_classes, epsilon=1e-6):
        """
        Initialize FedAlignLoss.
        
        Args:
            label_counts: Dictionary mapping each class label to its frequency count
            num_classes: Total number of classes in the dataset
            epsilon: Small constant for numerical stability
        """
        super(FedAlignLoss, self).__init__()
        self.label_counts = label_counts
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        # Calculate weights for each class based on label counts
        self.weights = torch.ones(num_classes)
        for label, count in label_counts.items():
            if 0 <= label < num_classes:  # Ensure label is valid
                # Inverse weighting with epsilon for stability
                self.weights[label] = 1.0 / (count + epsilon)
        
        # Normalize weights to sum to 1
        if self.weights.sum() > 0:
            self.weights = self.weights / self.weights.sum() * num_classes
        
    def forward(self, outputs, targets):
        """
        Calculate the reweighted loss.
        
        Args:
            outputs: Model predictions (logits)
            targets: Ground truth class indices
            
        Returns:
            weighted_loss: Class-balanced loss
        """
        # Convert weights to the same device as targets
        weights = self.weights.to(outputs.device)
        
        # Handle potential shape issues - check if we need to reshape outputs/targets
        if outputs.dim() > 2 and targets.dim() == 1:
            # Handle sequence output case - flatten the sequence dimension
            batch_size = outputs.size(0)
            seq_len = outputs.size(1)
            num_classes = outputs.size(2)
            # Reshape outputs to (batch_size * seq_len, num_classes)
            outputs = outputs.reshape(-1, num_classes)
            # In this case targets should be repeated
            if targets.size(0) == batch_size:
                # Repeat each target seq_len times
                targets = targets.repeat_interleave(seq_len)
        
        # Sanity check - verify batch sizes match
        if outputs.size(0) != targets.size(0):
            # Fallback to standard CrossEntropyLoss for safety
            print(f"Warning: FedAlignLoss - batch size mismatch: outputs {outputs.size(0)}, targets {targets.size(0)}. Using standard loss.")
            return F.cross_entropy(outputs, targets, reduction='mean')
        
        # Calculate cross-entropy loss with class weights
        criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')
        per_sample_losses = criterion(outputs, targets)
        
        # Calculate weighted average (can apply additional weighting per sample if needed)
        weighted_loss = per_sample_losses.mean()
        
        return weighted_loss 