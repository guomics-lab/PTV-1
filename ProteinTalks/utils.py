import torch

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path):
    """
    Save a model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        checkpoint_path: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def check_early_stopping(early_stopping, val_loss, epoch):
    """
    Check if early stopping criteria are met and update tracking variables

    Args:
        early_stopping: Dictionary with early stopping tracking variables
        val_loss: Current validation loss
        epoch: Current epoch

    Returns:
        Dictionary with results including:
          - early_stopping: Updated early_stopping dictionary
          - stop_training: Boolean indicating if training should stop
          - save_model: Boolean indicating if model should be saved
    """
    score = -val_loss  # Higher score is better
    save_model = False

    if early_stopping['best_score'] is None:
        # First epoch
        early_stopping['best_score'] = score
        save_model = True
    elif score < early_stopping['best_score'] + early_stopping['delta']:
        # Score didn't improve enough
        early_stopping['counter'] += 1
        print(f"EarlyStopping counter: {early_stopping['counter']} out of {early_stopping['patience']}")

    else:
        # Score improved, reset counter and save model
        early_stopping['best_score'] = score
        early_stopping['counter'] = 0
        save_model = True

    return {
        'early_stopping': early_stopping,
        'stop_training': early_stopping['counter'] >= early_stopping['patience'],
        'save_model': save_model
    }
