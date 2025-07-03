"""
Simple evaluation function for federated learning experiments.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import collate_batch

def evaluate(model, test_dataset, device, batch_size=64):
    """
    Evaluate a model on a test dataset.
    
    Args:
        model: The model to evaluate
        test_dataset: The test dataset
        device: The device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        test_loss: Average test loss
        test_accuracy: Test accuracy percentage
    """
    model.eval()
    model.to(device)
    
    # Check if it's a text dataset that needs custom collate function
    is_text_dataset = False
    if hasattr(test_dataset, 'vocab_size') and hasattr(test_dataset, 'num_classes'):
        # This is likely a text dataset (AG News, etc.)
        is_text_dataset = True
    
    # Create dataloader with appropriate collate function
    if is_text_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=collate_batch)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # Handle different batch formats
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                # Text data format: (labels, data, offsets)
                labels, data, offsets = batch_data
                data, labels, offsets = data.to(device), labels.to(device), offsets.to(device)
                outputs = model(data, offsets)
            elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                # Standard format: (data, labels)
                data, labels = batch_data
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
            else:
                continue
            
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / total if total > 0 else float('inf')
    test_accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_test_loss, test_accuracy 