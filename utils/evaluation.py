import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import parameters_to_vector

# Import the collate function to use with text datasets
try:
    from utils.data_loader import collate_batch
except ImportError:
    collate_batch = None
    print("Warning: collate_batch not imported in evaluation.py. Text datasets might not evaluate correctly.")

def evaluate_loss(model, dataloader, device):
    """Evaluate the loss on a validation/test dataloader."""
    model.eval()
    total_loss_sum = 0.0
    valid_samples_for_loss = 0 # Tracks samples that contributed to a valid loss
    batch_count = 0
    
    # Check dataset type
    dataset = dataloader.dataset
    is_potentially_text = hasattr(dataset, 'vocab_size')
    
    print(f"Starting evaluate_loss - dataloader has {len(dataloader)} batches")
    
    # --- Add check if dataloader is empty --- 
    try:
        first_batch = next(iter(dataloader))
        # Reset iterator if needed, though usually not necessary for evaluation
    except StopIteration:
        print("  CRITICAL: Dataloader is empty! No batches to evaluate.")
        return float('nan') # Return NaN if no data
    except Exception as e:
        print(f"  CRITICAL: Error retrieving first batch from dataloader: {e}")
        import traceback
        traceback.print_exc()
        return float('nan') # Return NaN on error
    # --- End check --- 
    
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            batch_count += 1
            try:
                # --- SIMPLIFIED Data Handling with Debug Prints ---
                # print(f"  Eval_loss batch {batch_idx}: Raw data_batch type: {type(data_batch)}")
                # if isinstance(data_batch, (list, tuple)):
                #     print(f"  Eval_loss batch {batch_idx}: Raw data_batch length: {len(data_batch)}")
                
                # Attempt to unpack standard case first
                if isinstance(data_batch, (list, tuple)) and len(data_batch) == 2:
                    # print(f"  Eval_loss batch {batch_idx}: Assuming standard (data, target) format.")
                    data, target = data_batch
                    # print(f"  Eval_loss batch {batch_idx}: Unpacked. Data type: {type(data)}, Target type: {type(target)}")
                    is_text_batch = False
                # Attempt to unpack text case 
                elif isinstance(data_batch, (list, tuple)) and len(data_batch) == 3:
                    # print(f"  Eval_loss batch {batch_idx}: Assuming text (target, data, offsets) format.")
                    target, data, offsets = data_batch # Note the order change here
                    # print(f"  Eval_loss batch {batch_idx}: Unpacked. Data type: {type(data)}, Target type: {type(target)}, Offsets type: {type(offsets)}")
                    is_text_batch = True
                else:
                    # print(f"  Eval_loss batch {batch_idx}: Unexpected data_batch format. Skipping.")
                    continue

                # Check for empty tensors immediately after unpacking
                if data.numel() == 0 or target.numel() == 0:
                    # print(f"  Eval_loss batch {batch_idx}: Skipping empty tensors after unpacking.")
                    continue
                
                # Move to device
                # print(f"  Eval_loss batch {batch_idx}: Moving data and target to device: {device}")
                data, target = data.to(device), target.to(device)
                if is_text_batch:
                    offsets = offsets.to(device)
                # --- END SIMPLIFIED Data Handling --- 
                
                # --- Model Forward Pass --- 
                # print(f"  Eval_loss batch {batch_idx}: Shapes before model - data={data.shape}, target={target.shape}")
                if is_text_batch:
                    output = model(data, offsets)
                else:
                    output = model(data)
                output_flat = output
                target_flat = target
                # print(f"  Eval_loss batch {batch_idx}: Output shape before flatten: {output.shape}")
                
                if len(output.shape) == 3:
                    current_vocab_size = output.size(-1)
                    output_flat = output.view(-1, current_vocab_size)
                    target_flat = target.view(-1)
                    # print(f"  Eval_loss batch {batch_idx}: Shapes after flatten - output_flat={output_flat.shape}, target_flat={target_flat.shape}")
                
                # Verify tensors are not empty before loss calculation
                if output_flat.numel() == 0 or target_flat.numel() == 0:
                    # print(f"  Eval_loss batch {batch_idx}: Skipping empty tensors")
                    continue
                
                # print(f"  Eval_loss batch {batch_idx}: Calculating loss...")
                loss = F.cross_entropy(output_flat, target_flat, reduction='sum')
                batch_loss = loss.item()
                batch_size = target_flat.numel()
                # print(f"  Eval_loss batch {batch_idx}: Batch loss={batch_loss:.4f}, Batch size={batch_size}")
                # --- End added prints --- 
                
                # Check for invalid loss
                if not np.isnan(batch_loss) and not np.isinf(batch_loss):
                    total_loss_sum += batch_loss # loss is already sum for the batch
                    valid_samples_for_loss += batch_size
                else:
                    print(f"Warning (evaluate_loss): Batch {batch_idx} produced invalid loss ({batch_loss}). Skipping from average.")
                
                # Debug: Print shapes and loss for the first batch for diagnostics
                if batch_idx == 0:
                    # print(f"Eval diag: data shape {data.shape}, target shape {target.shape}, device {device}")
                    # print(f"Eval diag: output shape {output.shape}, loss {loss.item()}")
                    pass # Pass if lines are commented
                
            except Exception as e:
                print(f"--- ERROR in evaluate_loss batch {batch_idx} ---")
                import traceback
                traceback.print_exc()
                raise e
    
    # Calculate average loss
    if valid_samples_for_loss > 0:
        avg_loss = total_loss_sum / valid_samples_for_loss
    else:
        avg_loss = float('nan') # Return NaN if no valid batches were processed
    print(f"Eval complete: total_loss_sum={total_loss_sum:.2f}, valid_samples_for_loss={valid_samples_for_loss}, avg_loss={avg_loss:.4f}")
    return avg_loss

def evaluate_accuracy(model, dataloader, device):
    """Evaluate the accuracy on a validation/test dataloader."""
    model.eval()
    correct = 0
    total = 0
    batch_count = 0
    
    # Check dataset type
    dataset = dataloader.dataset
    is_potentially_text = hasattr(dataset, 'vocab_size')
    
    print(f"Starting evaluate_accuracy - dataloader has {len(dataloader)} batches")
    
    # --- Add check if dataloader is empty --- 
    try:
        first_batch = next(iter(dataloader))
        # Reset iterator if needed
    except StopIteration:
        print("  CRITICAL: Dataloader is empty! No batches to evaluate.")
        return 0.0, {} # Return 0 accuracy and empty dict if no data
    except Exception as e:
        print(f"  CRITICAL: Error retrieving first batch from dataloader: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, {} # Return 0 accuracy and empty dict on error
    # --- End check --- 
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            batch_count += 1
            try:
                # --- SIMPLIFIED Data Handling with Debug Prints ---
                # print(f"  Eval_acc batch {batch_idx}: Raw data_batch type: {type(data_batch)}")
                # if isinstance(data_batch, (list, tuple)):
                #     print(f"  Eval_acc batch {batch_idx}: Raw data_batch length: {len(data_batch)}")

                # Attempt to unpack standard case first
                if isinstance(data_batch, (list, tuple)) and len(data_batch) == 2:
                    # print(f"  Eval_acc batch {batch_idx}: Assuming standard (data, target) format.")
                    data, target = data_batch
                    # print(f"  Eval_acc batch {batch_idx}: Unpacked. Data type: {type(data)}, Target type: {type(target)}")
                    is_text_batch = False
                # Attempt to unpack text case 
                elif isinstance(data_batch, (list, tuple)) and len(data_batch) == 3:
                    # print(f"  Eval_acc batch {batch_idx}: Assuming text (target, data, offsets) format.")
                    target, data, offsets = data_batch # Note the order change here
                    # print(f"  Eval_acc batch {batch_idx}: Unpacked. Data type: {type(data)}, Target type: {type(target)}, Offsets type: {type(offsets)}")
                    is_text_batch = True
                else:
                    # print(f"  Eval_acc batch {batch_idx}: Unexpected data_batch format. Skipping.")
                    continue

                # Check for empty tensors immediately after unpacking
                if data.numel() == 0 or target.numel() == 0:
                    # print(f"  Eval_acc batch {batch_idx}: Skipping empty tensors after unpacking.")
                    continue
                
                # Move to device
                # print(f"  Eval_acc batch {batch_idx}: Moving data and target to device: {device}")
                data, target = data.to(device), target.to(device)
                if is_text_batch:
                    offsets = offsets.to(device)
                # --- END SIMPLIFIED Data Handling --- 
                
                # --- Model Forward Pass --- 
                # print(f"  Eval_acc batch {batch_idx}: Shapes before model - data={data.shape}, target={target.shape}")
                if is_text_batch:
                    output = model(data, offsets)
                else:
                    output = model(data)
                output_flat = output
                target_flat = target
                # print(f"  Eval_acc batch {batch_idx}: Output shape before flatten: {output.shape}")
                
                if len(output.shape) == 3:
                    current_vocab_size = output.size(-1)
                    output_flat = output.view(-1, current_vocab_size)
                    target_flat = target.view(-1)
                    # print(f"  Eval_acc batch {batch_idx}: Shapes after flatten - output_flat={output_flat.shape}, target_flat={target_flat.shape}")
                
                # Verify tensors are not empty before accuracy calculation
                if output_flat.numel() == 0 or target_flat.numel() == 0:
                    # print(f"  Eval_acc batch {batch_idx}: Skipping empty tensors")
                    continue
                
                # print(f"  Eval_acc batch {batch_idx}: Calculating accuracy...")
                _, predicted = torch.max(output_flat.data, 1)
                batch_correct = (predicted == target_flat).sum().item()
                batch_total = target_flat.size(0)
                # print(f"  Eval_acc batch {batch_idx}: Correct={batch_correct}, Total={batch_total}")
                # --- End added prints --- 
                
                # Debug: Print accuracy for the first batch
                if batch_idx == 0:
                    # batch_correct = (predicted == target).sum().item() # This was part of the original user code, ensure it's correct if uncommented
                    # batch_total = target.size(0) # This was part of the original user code
                    # print(f"Acc eval diag: First batch correct {batch_correct}/{batch_total}")
                    pass # Pass if lines are commented
                
                # Update totals
                correct += batch_correct
                total += batch_total
                
                # Store batch predictions and targets for per-class accuracy
                all_targets.append(target_flat.cpu())
                all_predictions.append(predicted.cpu())
                    
            except Exception as e:
                print(f"--- ERROR in evaluate_accuracy batch {batch_idx} ---")
                import traceback
                traceback.print_exc()
                raise e
    
    # Calculate accuracy
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Accuracy eval complete: correct={correct}, total={total}, accuracy={accuracy:.2f}%")
    
    # Calculate per-class accuracy
    per_class_accuracy_map = {}
    if total > 0:
        all_targets_np = torch.cat(all_targets).numpy()
        all_predictions_np = torch.cat(all_predictions).numpy()
        
        num_classes = 0
        if hasattr(dataset, 'num_classes'):
            num_classes = dataset.num_classes
        elif hasattr(model, 'num_classes'): # Fallback to model attribute
            num_classes = model.num_classes
        elif hasattr(dataset, 'classes'): # Fallback for torchvision datasets
            num_classes = len(dataset.classes)
        else:
            # Try to infer from max label value if not too sparse
            if len(all_targets_np) > 0:
                num_classes = int(np.max(all_targets_np) + 1) 
            if num_classes == 0 or num_classes > 1000: # Safety check for large inferred num_classes
                 print(f"Warning: Could not reliably determine num_classes. Per-class accuracy might be incomplete or incorrect.")
                 # Default to a reasonable max or skip if num_classes is still 0
                 num_classes = max(1, num_classes) # Ensure num_classes is at least 1 if we proceed

        if num_classes > 0:
            correct_per_class = np.zeros(num_classes)
            total_per_class = np.zeros(num_classes)

            for i in range(len(all_targets_np)):
                true_label = all_targets_np[i]
                predicted_label = all_predictions_np[i]
                if true_label < num_classes and predicted_label < num_classes : # Ensure labels are within range
                    total_per_class[true_label] += 1
                    if true_label == predicted_label:
                        correct_per_class[true_label] += 1
            
            for c in range(num_classes):
                if total_per_class[c] > 0:
                    class_acc = 100.0 * correct_per_class[c] / total_per_class[c]
                else:
                    class_acc = 0.0 # Or float('nan')
                per_class_accuracy_map[c] = class_acc
                # print(f"Class {c} Accuracy: {class_acc:.2f}% ({int(correct_per_class[c])}/{int(total_per_class[c])})")
        else:
            print("Warning: num_classes is 0. Skipping per-class accuracy calculation.")

    return accuracy, per_class_accuracy_map

def parameter_distance(model_a, model_b):
    """Calculate Euclidean distance between parameters of two models."""
    distance = 0.0
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    
    for name, param_a in params_a.items():
        if name in params_b:
            param_b = params_b[name]
            distance += torch.norm(param_a - param_b).item() ** 2
            
    return distance ** 0.5

def early_stopping(val_losses, patience=5, delta=0):
    """Determine if training should stop early based on validation loss.
    Returns True if should stop, False otherwise.
    """
    if len(val_losses) < patience + 1:
        return False
    
    # Get the best (lowest) loss seen so far
    best_loss = min(val_losses[:-patience])
    
    # Check if validation loss hasn't improved for 'patience' epochs
    for i in range(-patience, 0):
        # If any loss in patience window is better than best loss minus delta, return False
        if val_losses[i] < best_loss - delta:
            return False
    
    # If we get here, loss hasn't improved for patience epochs
    return True

def create_lr_scheduler(optimizer, mode='step', **kwargs):
    """
    Create a learning rate scheduler for the optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        mode: Scheduler type ('step', 'cosine', 'plateau', 'exponential')
        **kwargs: Additional arguments specific to each scheduler type
            - step: step_size (epochs per decay), gamma (decay factor)
            - cosine: T_max (max iterations), eta_min (min lr)
            - plateau: factor (decay factor), patience (epochs to wait), threshold
            - exponential: gamma (decay factor)
            
    Returns:
        scheduler: PyTorch learning rate scheduler
    """
    if mode == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif mode == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif mode == 'plateau':
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        threshold = kwargs.get('threshold', 1e-4)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, threshold=threshold
        )
    
    elif mode == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")

def warmup_learning_rate(epoch, warmup_epochs, base_lr):
    """
    Calculate learning rate with linear warmup.
    
    Args:
        epoch: Current epoch number
        warmup_epochs: Number of epochs for warmup
        base_lr: Target learning rate after warmup
        
    Returns:
        lr: Learning rate for the current epoch
    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        return base_lr

def model_size(model):
    """Calculate the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def model_memory_usage(model):
    """Calculate the memory usage of a model in MB."""
    mem_params = sum([p.numel() * p.element_size() for p in model.parameters()])
    mem_bufs = sum([b.numel() * b.element_size() for b in model.buffers()])
    total_mem_bytes = mem_params + mem_bufs
    return total_mem_bytes / (1024**2) 