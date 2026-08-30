#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SWAG (Stochastic Weight Averaging-Gaussian) for ProteinTalks.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class SWAG(torch.optim.Optimizer):
    """
    SWAG (Stochastic Weight Averaging - Gaussian) optimizer

    This implements the SWAG algorithm for Bayesian deep learning as described in:
    "A Simple Baseline for Bayesian Uncertainty in Deep Learning" (Maddox et al., 2019)

    Args:
        base_optimizer: The base optimizer (e.g., SGD, Adam)
        swa_start: Epoch when SWAG collection starts
        swa_freq: Frequency of collecting SWAG snapshots (in epochs)
        swa_lr: Learning rate for SWAG phase (should be constant)
        max_num_models: Maximum number of models to store for covariance estimation
        var_clamp: Clamp for minimum variance values
    """

    def __init__(self, base_optimizer, swa_start=10, swa_freq=1, swa_lr=0.01,
                 max_num_models=20, var_clamp=1e-30):

        if not isinstance(base_optimizer, torch.optim.Optimizer):
            raise TypeError(f'base_optimizer must be an Optimizer, got {type(base_optimizer)}')

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults

        # SWAG specific parameters
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

        # SWAG state
        self.n_models = 0
        self.collected_models = []

        # Initialize SWAG state (will be done lazily on first update)
        self._swag_initialized = False

    def _initialize_swag_state(self):
        """
        Initialize SWAG state lazily
        """
        if self._swag_initialized:
            return

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # Ensure parameter has state entry
                    if p not in self.state:
                        self.state[p] = {}

                    self.state[p]['swag_mean'] = torch.zeros_like(p.data)
                    self.state[p]['swag_sq_mean'] = torch.zeros_like(p.data)

        self._swag_initialized = True

    def update_swag(self):
        """
        Update SWAG statistics with current model parameters
        """
        # Initialize SWAG state if not already done
        self._initialize_swag_state()

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    param_state = self.state[p]

                    # Update first moment (mean)
                    if self.n_models == 0:
                        param_state['swag_mean'].copy_(p.data)
                        param_state['swag_sq_mean'].copy_(p.data ** 2)
                    else:
                        # Running average
                        param_state['swag_mean'].mul_(self.n_models / (self.n_models + 1.0))
                        param_state['swag_mean'].add_(p.data / (self.n_models + 1.0))

                        # Running average of squared parameters
                        param_state['swag_sq_mean'].mul_(self.n_models / (self.n_models + 1.0))
                        param_state['swag_sq_mean'].add_((p.data ** 2) / (self.n_models + 1.0))

        # Store model for low-rank plus diagonal covariance
        if len(self.collected_models) < self.max_num_models:
            model_dict = OrderedDict()
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        model_dict[id(p)] = p.data.clone()
            self.collected_models.append(model_dict)
        else:
            # Replace oldest model (circular buffer)
            idx = self.n_models % self.max_num_models
            model_dict = OrderedDict()
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        model_dict[id(p)] = p.data.clone()
            self.collected_models[idx] = model_dict

        self.n_models += 1

    def sample(self, scale=1.0, cov=True, seed=None):
        """
        Sample from the SWAG posterior

        Args:
            scale: Scale factor for sampling
            cov: Whether to use covariance (low-rank) component
            seed: Random seed for sampling

        Returns:
            sampled_params: Dictionary of sampled parameters
        """
        if seed is not None:
            torch.manual_seed(seed)

        if self.n_models == 0:
            raise RuntimeError("No SWAG models collected yet")

        # Ensure SWAG state is initialized
        self._initialize_swag_state()

        sampled_params = OrderedDict()

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    param_state = self.state[p]

                    # Get mean and diagonal variance
                    mean = param_state['swag_mean']
                    sq_mean = param_state['swag_sq_mean']
                    var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

                    # Sample from diagonal Gaussian
                    eps = torch.randn_like(mean)
                    sample = mean + scale * torch.sqrt(var) * eps

                    # Add low-rank component if requested and available
                    if cov and len(self.collected_models) > 1:
                        # Compute deviation matrix
                        deviations = []
                        for model_dict in self.collected_models:
                            if id(p) in model_dict:
                                dev = (model_dict[id(p)] - mean).flatten()
                                deviations.append(dev)

                        if deviations:
                            deviation_matrix = torch.stack(deviations, dim=1)  # [param_dim, num_models]

                            # Sample from low-rank component
                            K = deviation_matrix.shape[1]
                            z = torch.randn(K, device=deviation_matrix.device)
                            low_rank_sample = torch.mv(deviation_matrix, z) / np.sqrt(K - 1)
                            low_rank_sample = low_rank_sample.view_as(mean)

                            sample = sample + scale * low_rank_sample

                    sampled_params[id(p)] = sample

        return sampled_params

    def set_swag_mode(self, model):
        """
        Set model parameters to SWAG mean

        Args:
            model: PyTorch model to update
        """
        if self.n_models == 0:
            return

        # Ensure SWAG state is initialized
        self._initialize_swag_state()

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    param_state = self.state[p]
                    p.data.copy_(param_state['swag_mean'])

    def set_sampled_mode(self, model, sampled_params):
        """
        Set model parameters to sampled values

        Args:
            model: PyTorch model to update
            sampled_params: Dictionary of sampled parameters from sample()
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and id(p) in sampled_params:
                    p.data.copy_(sampled_params[id(p)])

    def step(self, closure=None):
        """
        Perform a single optimization step

        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        # Set learning rate to SWAG learning rate
        for group in self.param_groups:
            group['lr'] = self.swa_lr

        # Perform base optimizer step
        loss = self.base_optimizer.step(closure)
        return loss

    def zero_grad(self):
        """
        Clear gradients
        """
        self.base_optimizer.zero_grad()

    def collect_model(self, epoch):
        """Collect a model at the configured fixed frequency."""
        if epoch < self.swa_start:
            return False, "before_swa_start"

        if (epoch - self.swa_start) % self.swa_freq == 0:
            self.update_swag()
            return True, "fixed_frequency"

        return False, "fixed_frequency_skip"

    def get_space_requirements(self):
        """
        Get memory requirements for SWAG

        Returns:
            dict: Memory requirements information
        """
        total_params = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    total_params += p.numel()

        # Each collected model stores a copy of all parameters
        model_memory = total_params * len(self.collected_models) * 4  # assuming float32

        # Mean and squared mean
        statistics_memory = total_params * 2 * 4  # float32

        return {
            'total_params': total_params,
            'collected_models': len(self.collected_models),
            'model_memory_mb': model_memory / (1024 * 1024),
            'statistics_memory_mb': statistics_memory / (1024 * 1024),
            'total_memory_mb': (model_memory + statistics_memory) / (1024 * 1024)
        }


def update_bn(loader, model, device=None):
    """
    Update BatchNorm statistics for SWAG model

    Args:
        loader: DataLoader for updating BN statistics
        model: Model with BatchNorm layers
        device: Device to use for computation
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()

    # Reset BatchNorm statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
        elif hasattr(module, 'reset_running_stats'):
            module.reset_running_stats()

    # Update statistics with limited number of batches to save time
    max_batches = min(len(loader), 100)  # Limit to 100 batches for efficiency

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            try:
                if isinstance(batch, (list, tuple)) and len(batch) >= 6:
                    # Neural ODE model expects: x, pert, fp_phA, fp_phB, time_stamp
                    x = batch[0].to(device)
                    pert = batch[1].to(device)
                    fp_phA = batch[4].to(device)
                    fp_phB = batch[5].to(device)

                    # Use 'all' as default time stamp for BN update
                    time_stamp = 'all'

                    # Forward pass to update BN stats
                    _ = model(x, pert, fp_phA, fp_phB, time_stamp)

                elif isinstance(batch, (list, tuple)):
                    # Fallback for simpler batch format
                    x = batch[0].to(device)
                    _ = model(x)
                else:
                    # Single tensor batch
                    x = batch.to(device)
                    _ = model(x)

            except Exception as e:
                print(f"Warning: Could not update BN stats for batch {i}: {e}")
                # Continue with next batch instead of breaking
                continue

    print(f"Updated BatchNorm statistics using {min(max_batches, len(loader))} batches")


class SWAGCallback:
    """
    Callback for managing SWAG training phase
    """

    def __init__(self, swag_optimizer, min_lr_factor=0.2):
        """Initialize the callback controlling the SWAG start condition."""
        self.swag_optimizer = swag_optimizer
        self.min_lr_factor = min_lr_factor
        self.swag_started = False
        self.initial_lr = None

    def should_start_swag(self, scheduler, epoch):
        """
        Check if SWAG should be started

        Args:
            scheduler: Learning rate scheduler
            epoch: Current epoch

        Returns:
            bool: Whether to start SWAG
        """
        if self.swag_started:
            return False

        if self.initial_lr is None:
            # Store initial learning rate
            if hasattr(scheduler, 'base_lrs'):
                self.initial_lr = scheduler.base_lrs[0]
            else:
                self.initial_lr = scheduler.optimizer.param_groups[0]['lr']

        # Check conditions for starting SWAG
        current_lr = scheduler.optimizer.param_groups[0]['lr']

        # Condition 1: Recent LR reduction + patience reset
        plateau_ready = (hasattr(scheduler, 'num_bad_epochs') and
                        scheduler.num_bad_epochs == 0)

        # Condition 2: LR dropped significantly
        lr_dropped = current_lr < self.initial_lr * self.min_lr_factor

        return plateau_ready and lr_dropped

    def start_swag(self, epoch):
        """
        Start SWAG phase

        Args:
            epoch: Current epoch
        """
        if not self.swag_started:
            print(f'Starting SWAG at epoch {epoch}, lr reset to {self.swag_optimizer.swa_lr}')

            # Reset learning rate to SWAG learning rate
            for group in self.swag_optimizer.param_groups:
                group['lr'] = self.swag_optimizer.swa_lr

            self.swag_started = True
            return True
        return False

    def step(self, epoch):
        """Collect a snapshot when the fixed-frequency condition is met."""
        if self.swag_started:
            return self.swag_optimizer.collect_model(epoch)[0]
        return False
