                    # --- Debug: Calculate Gradient Norm ---
                    total_norm = 0.0
                    for p in local_model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() # Use stored item
                    local_steps += 1
                    num_batches += 1 