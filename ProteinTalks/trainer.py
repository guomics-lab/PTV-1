import torch
import torch.nn as nn
import os
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
import torch.nn.functional as F
from utils import save_checkpoint, check_early_stopping
from swag import SWAG, SWAGCallback, update_bn
from metrics import calculate_detailed_metrics
from plot import (
    plot_roc_curve,
    plot_pr_curve
)

class Trainer:
    """
    Trainer class for model training and evaluation
    """
    def __init__(self, model, optimizer, scheduler, args, device):
        """
        Initialize the trainer

        Args:
            model: The model to train
            optimizer: The optimizer to use
            scheduler: The learning rate scheduler to use
            args: Command line arguments
            device: Device to use for training (CPU/GPU)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.device = device
        self.criterion = nn.BCELoss()  

        
        self.early_stopping = {
            'patience': args.patience,
            'counter': 0,
            'best_score': None,
            'early_stop': False,
            'delta': 0
        }

        
        self.swag_optimizer = None
        self.swag_callback = None
        if args.use_swag:
            print("Setting up SWAG...")
            self.swag_optimizer = SWAG(
                base_optimizer=optimizer,
                swa_start=0,  
                swa_freq=args.swag_freq,
                swa_lr=args.swag_lr,
                max_num_models=args.swag_max_models
            )
            self.swag_callback = SWAGCallback(
                swag_optimizer=self.swag_optimizer,
                min_lr_factor=args.swag_start_factor
            )

            print("SWAG initialized with fixed collection:")
            print(f"  lr={args.swag_lr}, freq={args.swag_freq}, max_models={args.swag_max_models}")

    def pearson_cor(self, outputs, targets):
        """Calculate Pearson correlation between predicted and actual protein expressions"""
        
        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        
        if len(outputs_np.shape) > 2:
            outputs_np = outputs_np.reshape(outputs_np.shape[0], -1)
            targets_np = targets_np.reshape(targets_np.shape[0], -1)

        
        correlations = []
        for i in range(outputs_np.shape[0]):
            corr = np.corrcoef(outputs_np[i], targets_np[i])[0, 1]
            correlations.append(corr)

        
        return np.mean(correlations)

    def train_epoch(self, train_dataloader):
        """
        Train for one epoch

        Args:
            train_dataloader: DataLoader for training data

        Returns:
            metrics: Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0
        total_pro_loss = 0
        total_pheno_loss = 0
        total_pro_corr = 0
        all_predictions = []
        all_labels = []
        num_batches = len(train_dataloader)

        
        current_optimizer = self.swag_optimizer if (self.swag_optimizer and
                                                    self.swag_callback and
                                                    self.swag_callback.swag_started) else self.optimizer

        for batch in train_dataloader:
            
            current_optimizer.zero_grad()

            
            x, pert, y, pheno, fp_phA, fp_phB = [b.to(self.device) for b in batch]

            
            outputs, pheno_pred, _ = self.model(x, pert, fp_phA, fp_phB, self.args.time_stamp_predict_drug)

            
            pro_loss = F.mse_loss(outputs, y)
            pheno_loss = self.criterion(pheno_pred, pheno)

            
            pro_corr = self.pearson_cor(outputs, y)

            
            loss = (1 - self.args.lambda_pheno) * pro_loss + self.args.lambda_pheno * pheno_loss

            
            loss.backward()

            
            current_optimizer.step()

            
            total_loss += loss.item()
            total_pro_loss += pro_loss.item()
            total_pheno_loss += pheno_loss.item()
            total_pro_corr += pro_corr
            all_predictions.extend(pheno_pred.detach().cpu().numpy())
            all_labels.extend(pheno.detach().cpu().numpy())

        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        
        all_labels = (all_labels > 0.5).astype(int)
        binary_predictions = (all_predictions > 0.5).astype(int)

        accuracy = accuracy_score(all_labels, binary_predictions)

        
        fpr, tpr, _ = roc_curve(all_labels, all_predictions, pos_label=1)
        auroc = auc(fpr, tpr)

        
        precision, recall, _ = precision_recall_curve(all_labels, all_predictions, pos_label=1)
        auprc = average_precision_score(all_labels, all_predictions)

        metrics = {
            'loss': total_loss / num_batches,
            'pro_loss': total_pro_loss / num_batches,
            'pheno_loss': total_pheno_loss / num_batches,
            'pro_corr': total_pro_corr / num_batches,
            'accuracy': accuracy,
            'pheno_auprc': auprc,
            'pheno_auroc': auroc
        }

        return metrics

    def validate(self, validation_dataloader):
        """
        Validate the model

        Args:
            validation_dataloader: DataLoader for validation data

        Returns:
            metrics: Dictionary containing validation metrics
            predictions: Dictionary containing predictions and related data
        """
        self.model.eval()

        total_loss = 0
        total_pro_loss = 0
        total_pheno_loss = 0
        total_pro_corr = 0
        all_predictions = []
        all_labels = []
        num_batches = len(validation_dataloader)

        with torch.no_grad():
            for batch in validation_dataloader:
                
                x, pert, y, pheno, fp_phA, fp_phB = [b.to(self.device) for b in batch]

                
                outputs, pheno_pred, _ = self.model(x, pert, fp_phA, fp_phB, self.args.time_stamp_predict_drug)

                
                pro_loss = F.mse_loss(outputs, y)
                pheno_loss = self.criterion(pheno_pred, pheno)

                
                pro_corr = self.pearson_cor(outputs, y)
                
                loss = (1 - self.args.lambda_pheno) * pro_loss + self.args.lambda_pheno * pheno_loss

                
                total_loss += loss.item()
                total_pro_loss += pro_loss.item()
                total_pheno_loss += pheno_loss.item()
                total_pro_corr += pro_corr
                all_predictions.extend(pheno_pred.cpu().numpy())
                all_labels.extend(pheno.cpu().numpy())

        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        
        binary_predictions = (all_predictions >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(all_labels, all_predictions)
        auroc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        auprc = average_precision_score(all_labels, all_predictions)

        accuracy = accuracy_score(all_labels, binary_predictions)
        prec = precision_score(all_labels, binary_predictions, zero_division=0)
        rec = recall_score(all_labels, binary_predictions, zero_division=0)
        f1 = f1_score(all_labels, binary_predictions, zero_division=0)
        mcc = matthews_corrcoef(all_labels, binary_predictions)
        kappa = cohen_kappa_score(all_labels, binary_predictions)

        metrics = {
            'loss': total_loss / num_batches,
            'pro_loss': total_pro_loss / num_batches,
            'pheno_loss': total_pheno_loss / num_batches,
            'pro_corr': total_pro_corr / num_batches,
            'accuracy': accuracy,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'mcc': mcc,
            'kappa': kappa,
            'pheno_auroc': auroc,
            'pheno_auprc': auprc
        }

        predictions = {
            'predictions': all_predictions,
            'labels': all_labels,
            'binary_predictions': binary_predictions,
            'fpr': fpr,
            'tpr': tpr,
            'auroc': auroc,
            'precision': precision,
            'recall': recall,
            'auprc': auprc
        }

        return metrics, predictions

    def _save_validation_results(self, metrics, epoch):
        """
        Save validation results

        Args:
            metrics: Dictionary of validation metrics
            epoch: Current epoch
        """
        
        val_loss = metrics['pheno_loss']
        early_stopping_result = check_early_stopping(
            self.early_stopping, val_loss, epoch
        )

        
        if early_stopping_result['save_model']:
            best_checkpoint_path = os.path.join(
                self.args.dir_save,
                f"{epoch}_best_checkpoint.pth"
            )
            save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch, best_checkpoint_path
            )

        self.early_stopping = early_stopping_result['early_stopping']

        return early_stopping_result['stop_training']

    def train(self, train_dataloader, validation_dataloader, test_dataloader=None):
        """
        Train the model with SWAG support

        Args:
            train_dataloader: DataLoader for training data
            validation_dataloader: DataLoader for validation data
            test_dataloader: Optional DataLoader for test data

        Returns:
            Dictionary containing the best validation metrics
        """
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        best_val_metrics = None
        best_train_metrics = None
        patience_counter = 0

        print("Starting training...")

        
        swag_started = False
        current_optimizer = self.optimizer

        for epoch in range(self.args.total_epoch):
            
            if self.args.use_swag and self.swag_callback and not swag_started:
                
                if self.swag_callback.should_start_swag(self.scheduler, epoch):
                    swag_started = self.swag_callback.start_swag(epoch)
                    if swag_started:
                        current_optimizer = self.swag_optimizer
                        
                        self.swag_optimizer.swa_start = epoch
                        print(f"SWAG phase started at epoch {epoch}")

                        
                        mem_info = self.swag_optimizer.get_space_requirements()
                        print(f"SWAG memory requirements: {mem_info['total_memory_mb']:.1f} MB")

            
            train_metrics = self.train_epoch(train_dataloader)

            
            val_metrics, _ = self.validate(validation_dataloader)

            
            if swag_started:
                collected = self.swag_callback.step(epoch)
                if collected:
                    print(f"Collected SWAG snapshot {self.swag_optimizer.n_models} at epoch {epoch}")
            else:
                self.scheduler.step(val_metrics['loss'])

           
            patience_limit = self.args.patience * 2 if swag_started else self.args.patience

            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                best_model_state = deepcopy(self.model.state_dict())
                best_val_metrics = val_metrics
                best_train_metrics = train_metrics
                patience_counter = 0  

                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_metrics': val_metrics,
                    'swag_started': swag_started,
                }

                
                if swag_started and self.swag_optimizer:
                    checkpoint['swag_state'] = {
                        'n_models': self.swag_optimizer.n_models,
                        'swag_mean': {id(p): self.swag_optimizer.state[p]['swag_mean'].clone()
                                     for group in self.swag_optimizer.param_groups
                                     for p in group['params'] if p.requires_grad},
                        'swag_sq_mean': {id(p): self.swag_optimizer.state[p]['swag_sq_mean'].clone()
                                        for group in self.swag_optimizer.param_groups
                                        for p in group['params'] if p.requires_grad},
                        'collected_models': [model.copy() for model in self.swag_optimizer.collected_models]
                    }

                torch.save(checkpoint, os.path.join(self.args.dir_save, 'best_checkpoint.pt'))
            else:
                patience_counter += 1

                
                if patience_counter >= patience_limit:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

            
            if epoch % 200 == 0:
                current_lr = current_optimizer.param_groups[0]['lr']
                phase = "SWAG" if swag_started else "Normal"
                print(f"\nEpoch {epoch}/{self.args.total_epoch} [{phase} Phase]")
                print(f"Current LR: {current_lr:.2e}")
                self._print_status(epoch, train_metrics, val_metrics)
                print(f"Patience counter: {patience_counter}/{patience_limit}")

                if swag_started and self.swag_optimizer:
                    print(f"SWAG models collected: {self.swag_optimizer.n_models}")

        print(f"\nTraining completed. Best model was from epoch {best_epoch}")
        print("\nBest Model Performance:")
        self._print_status(best_epoch, best_train_metrics, best_val_metrics)

        
        if swag_started and self.swag_optimizer and self.swag_optimizer.n_models > 0:
            print(f"\nSWAG training completed with {self.swag_optimizer.n_models} collected models")

            
            print("Setting model to SWAG mean...")
            self.swag_optimizer.set_swag_mode(self.model)

            
            print("Updating BatchNorm statistics...")
            update_bn(train_dataloader, self.model, self.device)

            
            print("Evaluating SWAG mean model...")
            val_metrics_swag, _ = self.validate(validation_dataloader)
            print("SWAG Mean Model Performance:")
            self._print_validation_results(val_metrics_swag)

            
            swag_checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': self.model.state_dict(),
                'swag_optimizer_state': {
                    'n_models': self.swag_optimizer.n_models,
                    'swag_mean': {id(p): self.swag_optimizer.state[p]['swag_mean'].clone()
                                 for group in self.swag_optimizer.param_groups
                                 for p in group['params'] if p.requires_grad},
                    'swag_sq_mean': {id(p): self.swag_optimizer.state[p]['swag_sq_mean'].clone()
                                    for group in self.swag_optimizer.param_groups
                                    for p in group['params'] if p.requires_grad},
                    'collected_models': [model.copy() for model in self.swag_optimizer.collected_models]
                },
                'val_metrics': val_metrics_swag,
            }
            torch.save(swag_checkpoint, os.path.join(self.args.dir_save, 'swag_mean_checkpoint.pt'))

            
            if test_dataloader is not None:
                print("Evaluating SWAG with uncertainty estimation on test set...")
                self._evaluate_swag_uncertainty(test_dataloader)

        
        if test_dataloader is not None:
            print("\nEvaluating best deterministic model on test set...")
            self.model.load_state_dict(best_model_state)
            test_metrics, test_predictions = self.validate(test_dataloader)
            print("\nTest Results (Best Deterministic Checkpoint):")
            self._print_test_results(test_metrics)
            self._save_test_results(test_metrics, test_predictions)

        
        self._save_validation_results(best_val_metrics, best_epoch)

        return best_val_metrics

    def _evaluate_swag_uncertainty(self, test_dataloader):
        """
        Evaluate SWAG model with uncertainty estimation

        Args:
            test_dataloader: DataLoader for test data
        """
        if not self.swag_optimizer or self.swag_optimizer.n_models == 0:
            print("No SWAG models available for uncertainty estimation")
            return

        self.model.eval()
        num_samples = min(self.args.swag_samples, self.swag_optimizer.n_models * 2)

        
        all_sample_predictions = []
        all_sample_pro_outputs = []
        all_labels = []
        all_pro_targets = []

        print(f"Generating {num_samples} SWAG samples for uncertainty estimation...")

        for sample_idx in range(num_samples):
            
            sampled_params = self.swag_optimizer.sample(scale=1.0, cov=True, seed=sample_idx)
            self.swag_optimizer.set_sampled_mode(self.model, sampled_params)

            
            if sample_idx == 0:  
                update_bn(test_dataloader, self.model, self.device)

            
            sample_predictions = []
            sample_pro_outputs = []

            with torch.no_grad():
                for batch in test_dataloader:
                    x, pert, y, pheno, fp_phA, fp_phB = [b.to(self.device) for b in batch]

                    
                    outputs, pheno_pred, _ = self.model(x, pert, fp_phA, fp_phB,
                                                       self.args.time_stamp_predict_drug)

                    sample_predictions.extend(pheno_pred.cpu().numpy())
                    sample_pro_outputs.append(outputs.cpu().numpy())

                    
                    if sample_idx == 0:
                        all_labels.extend(pheno.cpu().numpy())
                        all_pro_targets.append(y.cpu().numpy())

            all_sample_predictions.append(np.array(sample_predictions))
            all_sample_pro_outputs.append(np.concatenate(sample_pro_outputs, axis=0))

        
        all_sample_predictions = np.array(all_sample_predictions)  
        all_sample_pro_outputs = np.array(all_sample_pro_outputs)  
        all_labels = np.array(all_labels)
        all_pro_targets = np.concatenate(all_pro_targets, axis=0)

        
        mean_predictions = np.mean(all_sample_predictions, axis=0)
        std_predictions = np.std(all_sample_predictions, axis=0)
        mean_pro_outputs = np.mean(all_sample_pro_outputs, axis=0)
        std_pro_outputs = np.std(all_sample_pro_outputs, axis=0)

        
        binary_predictions = (mean_predictions >= 0.5).astype(int)
        accuracy = accuracy_score(all_labels, binary_predictions)

        try:
            fpr, tpr, _ = roc_curve(all_labels, mean_predictions)
            auroc = auc(fpr, tpr)
        except ValueError as exc:
            warnings.warn(f"AUROC could not be computed: {exc}", RuntimeWarning)
            auroc = np.nan

        try:
            precision, recall, _ = precision_recall_curve(all_labels, mean_predictions)
            auprc = average_precision_score(all_labels, mean_predictions)
        except ValueError as exc:
            warnings.warn(f"AUPRC could not be computed: {exc}", RuntimeWarning)
            auprc = np.nan

        
        pro_corr = self.pearson_cor(torch.tensor(mean_pro_outputs), torch.tensor(all_pro_targets))

        
        print("\nSWAG Uncertainty Estimation Results:")
        print("Phenotype Prediction:")
        print(f"  Mean Accuracy: {accuracy:.4f}")
        print(f"  Mean AUROC: {auroc:.4f}")
        print(f"  Mean AUPRC: {auprc:.4f}")
        print(f"  Prediction Uncertainty (std): {np.mean(std_predictions):.4f}")
        print("Protein Expression:")
        print(f"  Mean Correlation: {pro_corr:.4f}")
        print(f"  Output Uncertainty (mean std): {np.mean(std_pro_outputs):.4f}")

        
        uncertainty_results = {
            'mean_predictions': mean_predictions,
            'std_predictions': std_predictions,
            'all_sample_predictions': all_sample_predictions,
            'mean_pro_outputs': mean_pro_outputs,
            'std_pro_outputs': std_pro_outputs,
            'ground_truth': all_labels,
            'pro_targets': all_pro_targets,
            'metrics': {
                'accuracy': accuracy,
                'auroc': auroc,
                'auprc': auprc,
                'pro_corr': pro_corr,
                'mean_uncertainty': np.mean(std_predictions),
                'mean_pro_uncertainty': np.mean(std_pro_outputs)
            }
        }

        
        uncertainty_path = os.path.join(self.args.dir_save, "swag_uncertainty_results.npz")
        np.savez(uncertainty_path, **uncertainty_results)

        
        uncertainty_metrics_path = os.path.join(self.args.dir_save, "swag_uncertainty_metrics.txt")
        with open(uncertainty_metrics_path, 'w') as f:
            f.write("SWAG Uncertainty Estimation Results\n")
            f.write("="*40 + "\n")
            f.write(f"Number of samples: {num_samples}\n")
            f.write(f"Number of collected SWAG models: {self.swag_optimizer.n_models}\n\n")

            f.write("Phenotype Prediction:\n")
            f.write(f"  Mean Accuracy: {accuracy:.4f}\n")
            f.write(f"  Mean AUROC: {auroc:.4f}\n")
            f.write(f"  Mean AUPRC: {auprc:.4f}\n")
            f.write(f"  Prediction Uncertainty (std): {np.mean(std_predictions):.4f}\n\n")

            f.write("Protein Expression:\n")
            f.write(f"  Mean Correlation: {pro_corr:.4f}\n")
            f.write(f"  Output Uncertainty (mean std): {np.mean(std_pro_outputs):.4f}\n")

        print(f"Uncertainty results saved to {uncertainty_path}")
        print(f"Uncertainty metrics saved to {uncertainty_metrics_path}")

        
        self.swag_optimizer.set_swag_mode(self.model)

    def _print_status(self, epoch, train_metrics, val_metrics):
        """Print training status"""
        print(
            f"Epoch {epoch}: "
            f"Train Loss: {train_metrics['loss']:.3f}, "
            f"Val Loss: {val_metrics['loss']:.3f}, "
            f"Train Pro Corr: {train_metrics['pro_corr']:.3f}, "
            f"Val Pro Corr: {val_metrics['pro_corr']:.3f}, "
            f"Train Pro Loss: {train_metrics['pro_loss']:.3f}, "
            f"Val Pro Loss: {val_metrics['pro_loss']:.3f}, "
            f"Train Pheno Loss: {train_metrics['pheno_loss']:.3f}, "
            f"Val Pheno Loss: {val_metrics['pheno_loss']:.3f}, "
            f"Train Pheno Acc: {train_metrics['accuracy']:.3f}, "
            f"Val Pheno Acc: {val_metrics['accuracy']:.3f}, "
            f"Train AUPRC: {train_metrics['pheno_auprc']:.3f}, "
            f"Val AUPRC: {val_metrics['pheno_auprc']:.3f}, "
            f"Train AUROC: {train_metrics['pheno_auroc']:.3f}, "
            f"Val AUROC: {val_metrics['pheno_auroc']:.3f}"
        )

    def _print_test_results(self, metrics):
        """Print test results"""
        print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Test Protein Loss: {metrics['pro_loss']:.4f}")
        print(f"Test Phenotype Loss: {metrics['pheno_loss']:.4f}")
        print(f"Test Protein Correlation: {metrics['pro_corr']:.4f}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test AUROC: {metrics['pheno_auroc']:.4f}")
        print(f"Test AUPRC: {metrics['pheno_auprc']:.4f}")

    def _save_test_results(self, metrics, predictions):
        """Save test results"""
        
        binary_predictions = (predictions['predictions'] >= 0.5).astype(int)
        binary_labels = predictions['labels'].astype(int)

        
        detailed_metrics = calculate_detailed_metrics(
            torch.tensor(binary_predictions),
            torch.tensor(binary_labels)
        )

        
        metrics_path = os.path.join(self.args.dir_save, "test_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Test Loss: {metrics['loss']:.4f}\n")
            f.write(f"Test Protein Loss: {metrics['pro_loss']:.4f}\n")
            f.write(f"Test Phenotype Loss: {metrics['pheno_loss']:.4f}\n")
            f.write(f"Test Protein Correlation: {metrics['pro_corr']:.4f}\n")
            f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Test AUROC: {metrics['pheno_auroc']:.4f}\n")
            f.write(f"Test AUPRC: {metrics['pheno_auprc']:.4f}\n")

            f.write("\nDetailed Metrics:\n")
            f.write(f"Precision: {detailed_metrics['precision']:.4f}\n")
            f.write(f"Recall: {detailed_metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {detailed_metrics['f1']:.4f}\n")
            f.write(f"Matthews Correlation Coefficient: {detailed_metrics['mcc']:.4f}\n")
            f.write(f"Cohen's Kappa: {detailed_metrics['kappa']:.4f}\n")

        
        roc_path = os.path.join(self.args.dir_save, "roc_curve_data.npz")
        np.savez(roc_path, fpr=predictions['fpr'], tpr=predictions['tpr'], auc=metrics['pheno_auroc'])

        
        pr_path = os.path.join(self.args.dir_save, "pr_curve_data.npz")
        np.savez(pr_path, precision=predictions['precision'], recall=predictions['recall'], auc=metrics['pheno_auprc'])

        
        results_path = os.path.join(self.args.dir_save, "test_predictions.npz")
        np.savez(
            results_path,
            ground_truth=predictions['labels'],
            predicted_probabilities=predictions['predictions'],
            binary_predictions=binary_predictions
        )

        
        results_df_path = os.path.join(self.args.dir_save, "test_predictions.csv")
        pd.DataFrame({
            'ground_truth': predictions['labels'],
            'predicted_probabilities': predictions['predictions'],
            'binary_predictions': binary_predictions
        }).to_csv(results_df_path, index=False)

        
        plot_roc_curve(predictions['fpr'], predictions['tpr'], metrics['pheno_auroc'], self.args.dir_save)
        plot_pr_curve(predictions['precision'], predictions['recall'], metrics['pheno_auprc'], self.args.dir_save)

    def _print_validation_results(self, metrics):
        """Print validation results"""
        print(f"Validation Loss: {metrics['loss']:.4f}")
        print(f"Validation Protein Loss: {metrics['pro_loss']:.4f}")
        print(f"Validation Phenotype Loss: {metrics['pheno_loss']:.4f}")
        print(f"Validation Protein Correlation: {metrics['pro_corr']:.4f}")
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation AUROC: {metrics['pheno_auroc']:.4f}")
        print(f"Validation AUPRC: {metrics['pheno_auprc']:.4f}")


def load_model_from_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Load model, optimizer, and scheduler from checkpoint

    Args:
        model: Model to load parameters into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to

    Returns:
        tuple: (model, optimizer, scheduler, epoch)
    """
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    
    model.load_state_dict(checkpoint['model_state_dict'])

    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    
    epoch = checkpoint.get('epoch', 0)

    return model, optimizer, scheduler, epoch
