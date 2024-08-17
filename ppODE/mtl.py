import torch
from torch.nn.utils import clip_grad_norm_

def compute_cosine_similarity(grads1, grads2):
    """
    calculated the cos similarity of two grad
    """
    # filter the grad without none
    valid_grads1 = [g.view(-1) for g, h in zip(grads1, grads2) if g is not None and h is not None]
    valid_grads2 = [h.view(-1) for g, h in zip(grads1, grads2) if g is not None and h is not None]

    if not valid_grads1 or not valid_grads2:
        return 0.0

    # merge the gradient
    valid_grads1 = torch.cat(valid_grads1)
    valid_grads2 = torch.cat(valid_grads2)

    # calculate cos similarity
    similarity = torch.nn.functional.cosine_similarity(valid_grads1.unsqueeze(0), valid_grads2.unsqueeze(0), dim=1)
    return similarity.item()

def adjust_weights_based_on_similarity(similarity, initial_weights, adjustment_rate=0.1):
    """
    Adjust task weights based on gradient similarity.
    Args:
        similarity: The similarity measure between two task gradients.
        initial_weights: Initial task weights (a list or tuple, e.g., [weight1, weight2]).
        adjustment_rate: The rate of adjustment, which determines the magnitude of weight adjustment.
    Returns:
        Adjusted task weights.
    """
    weight_task1, weight_task2 = initial_weights

    # if the gradient directions are similar
    if similarity > 0:
        # keep the weights unchanged
        pass

    # if the gradient directions are opposite
    elif similarity < 0:
        # modify the weights
        # # option1: (consider the fairness) increase the weight of the task with smaller weight, and decrease the weight of the task with larger weight
        # if weight_task1 > weight_task2:
        #     weight_task1 -= adjustment_rate * abs(similarity)
        #     weight_task2 += adjustment_rate * abs(similarity)
        # else:
        #     weight_task1 += adjustment_rate * abs(similarity)
        #     weight_task2 -= adjustment_rate * abs(similarity)
        # option2: (consider the importance of the task) increase the weight of important task, and decrease the weight of less important task
        weight_task1 += adjustment_rate * abs(similarity)
        weight_task2 -= adjustment_rate * abs(similarity)
        
    # clip the weights to the range [0, 1]
    weight_task1 = max(min(weight_task1, 1.0), 0.0)
    weight_task2 = max(min(weight_task2, 1.0), 0.0)

    return weight_task1, weight_task2

# def grad_clip(grad_task, clip_value):
#     clip_value = 1.0  
#     for grad in grad_task:
#         clip_grad_norm_(grad, clip_value)
#     return grad_task

def grad_clip(grads, clip_value):
    """
    Clip gradients element-wise by the given value. Ignore None gradients. Keep the original shape of gradients.
    """
    clipped_grads = []
    for grad in grads:
        if grad is not None:
            # clone the gradient to keep the original shape
            grad_clone = grad.clone()
            # clip the gradient element-wise, then reshape it back to the original shape
            grad_clone = grad_clone.reshape(-1)
            clip_grad_norm_(grad_clone, clip_value)
            grad_clone = grad_clone.view_as(grad)
            clipped_grads.append(grad_clone)
        else:
            clipped_grads.append(None)
    return clipped_grads