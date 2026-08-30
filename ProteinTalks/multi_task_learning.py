import torch
from torch import autograd
from torch.nn.utils import clip_grad_norm_

def compute_cosine_similarity(grads1, grads2):
    """
    Calculate the cosine similarity between two sets of gradients,
    considering only gradients that are not None in both sets.

    Args:
        grads1: First set of gradients
        grads2: Second set of gradients

    Returns:
        Cosine similarity as a float
    """
    # Filter out gradients that are None in either set
    valid_grads1 = [g.view(-1) for g, h in zip(grads1, grads2) if g is not None and h is not None]
    valid_grads2 = [h.view(-1) for g, h in zip(grads1, grads2) if g is not None and h is not None]

    # If no valid gradient pairs, return 0
    if not valid_grads1 or not valid_grads2:
        return 0.0

    # Concatenate gradients
    valid_grads1 = torch.cat(valid_grads1)
    valid_grads2 = torch.cat(valid_grads2)

    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        valid_grads1.unsqueeze(0),
        valid_grads2.unsqueeze(0),
        dim=1
    )

    return similarity.item()

def adjust_weights_based_on_similarity(similarity, initial_weights, adjustment_rate=0.1):
    """
    Adjust task weights based on gradient similarity.

    Args:
        similarity: Cosine similarity between gradients
        initial_weights: Initial weights for the tasks
        adjustment_rate: Rate at which to adjust weights

    Returns:
        Tuple of adjusted weights
    """
    weight_task1, weight_task2 = initial_weights

    # If gradients are similar or aligned, no adjustment needed
    if similarity > 0:
        pass  # Keep weights unchanged

    # If gradients conflict (negative similarity)
    elif similarity < 0:
        # Decrease weight for task 1, increase for task 2
        # This prioritizes drug efficacy/synergy prediction over protein reconstruction
        weight_task1 -= adjustment_rate * abs(similarity)
        weight_task2 += adjustment_rate * abs(similarity)

    # Keep weights in valid range [0, 1]
    weight_task1 = max(min(weight_task1, 1.0), 0.0)
    weight_task2 = max(min(weight_task2, 1.0), 0.0)

    # Renormalize so that the two task weights sum to one
    total = weight_task1 + weight_task2
    if total > 0:
        weight_task1 /= total
        weight_task2 /= total

    return weight_task1, weight_task2

def grad_clip(grads, clip_value):
    """
    Clip gradient norms while preserving their original shape.

    Args:
        grads: List of gradients
        clip_value: Maximum allowed gradient norm

    Returns:
        List of clipped gradients
    """
    clipped_grads = []
    for grad in grads:
        if grad is not None:
            grad_clone = grad.clone()
            # Reshape, clip, and restore shape
            grad_clone = grad_clone.reshape(-1)
            clip_grad_norm_(grad_clone, clip_value)
            grad_clone = grad_clone.view_as(grad)
            clipped_grads.append(grad_clone)
        else:
            clipped_grads.append(None)

    return clipped_grads

def calculate_task_gradients(model, loss_task1, loss_task2):
    """
    Calculate gradients for each task separately

    Args:
        model: The neural network model
        loss_task1: Loss for the first task (protein prediction)
        loss_task2: Loss for the second task (phenotype prediction)

    Returns:
        grad_task1: Gradients for first task
        grad_task2: Gradients for second task
    """
    # Calculate gradients for first task
    grad_task1 = autograd.grad(loss_task1, model.parameters(), retain_graph=True, allow_unused=True)

    # Calculate gradients for second task
    grad_task2 = autograd.grad(loss_task2, model.parameters(), allow_unused=True)

    return grad_task1, grad_task2

def apply_gradients(model, grad_task1, grad_task2, weight_task1, weight_task2):
    """
    Apply weighted gradients to model parameters

    Args:
        model: The neural network model
        grad_task1: Gradients for first task
        grad_task2: Gradients for second task
        weight_task1: Weight for first task
        weight_task2: Weight for second task
    """
    for param, g1, g2 in zip(model.parameters(), grad_task1, grad_task2):
        if param.grad is None:
            param.grad = torch.zeros_like(param)

        if g1 is not None and g2 is not None:
            # Check that shapes match
            if g1.size() == g2.size():
                param.grad += weight_task1 * g1 + weight_task2 * g2
            else:
                raise RuntimeError("Gradient size mismatch")
        elif g1 is not None:
            param.grad += weight_task1 * g1
        elif g2 is not None:
            param.grad += weight_task2 * g2

def multitask_step(model, optimizer, loss_task1, loss_task2, lambda_pheno=0.8, adjustment_rate=0.01):
    """
    Perform a multi-task learning step

    Args:
        model: The neural network model
        optimizer: The optimizer
        loss_task1: Loss for the first task (protein prediction)
        loss_task2: Loss for the second task (phenotype prediction)
        lambda_pheno: Initial weight for the phenotype task
        adjustment_rate: Rate at which to adjust weights

    Returns:
        combined_loss: The weighted combination of task losses
        weight_task1: Final weight for task 1
        weight_task2: Final weight for task 2
    """
    # Reset gradients
    optimizer.zero_grad()

    # Calculate gradients for each task
    grad_task1, grad_task2 = calculate_task_gradients(model, loss_task1, loss_task2)

    # Clip gradients
    clip_value = 1.0
    grad_task1 = grad_clip(grad_task1, clip_value)
    grad_task2 = grad_clip(grad_task2, clip_value)

    # Calculate gradient similarity
    similarity = compute_cosine_similarity(grad_task1, grad_task2)

    # Adjust weights based on similarity
    initial_weights = [1-lambda_pheno, lambda_pheno]
    weight_task1, weight_task2 = adjust_weights_based_on_similarity(
        similarity,
        initial_weights,
        adjustment_rate
    )

    # Calculate combined loss
    combined_loss = weight_task1 * loss_task1 + weight_task2 * loss_task2

    # Apply gradients
    apply_gradients(model, grad_task1, grad_task2, weight_task1, weight_task2)

    # Update parameters
    optimizer.step()

    return combined_loss, weight_task1, weight_task2
