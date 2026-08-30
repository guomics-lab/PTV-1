import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    accuracy_score,
)


def calculate_detailed_metrics(output, target):
    """Calculate classification metrics for phenotype prediction."""
    output_np = output.detach().cpu().numpy() if torch.is_tensor(output) else output
    target_np = target.detach().cpu().numpy() if torch.is_tensor(target) else target

    output_binary = (output_np >= 0.5).astype(int)
    target_binary = target_np.astype(int)

    return {
        'accuracy': accuracy_score(target_binary, output_binary),
        'precision': precision_score(target_binary, output_binary, zero_division=0),
        'recall': recall_score(target_binary, output_binary, zero_division=0),
        'f1': f1_score(target_binary, output_binary, zero_division=0),
        'mcc': matthews_corrcoef(target_binary, output_binary),
        'kappa': cohen_kappa_score(target_binary, output_binary),
    }
