#!/usr/bin/env python

"""
Main entry point for ProteinTalks proteomic and phenotype prediction.
"""

import os
import sys
import torch
import numpy as np
import pickle
import pandas as pd

from config import get_args, setup_device, setup_directories
from dataset import prepare_data, prepare_testdata
from model import ppODE
from trainer import Trainer, load_model_from_checkpoint

def main():
    """
    Main entry point for the application
    """
    
    args = get_args()

    
    device = setup_device(args)
    print(f"Using device: {device}")

    
    args.dir_save = setup_directories(args)
    print(f"Saving results to: {args.dir_save}")

    
    if args.train_from_scratch == "predict":
        
        if not args.cp_save_dir_best:
            raise ValueError("For prediction mode, --cp_save_dir_best must be specified")
        if not args.test_file_prefix:
            raise ValueError("For prediction mode, --test_file_prefix must be specified")

        print("Running in prediction mode...")
        print("Loading test data...")
        test_dataloader, pos_percent_info = prepare_testdata(args)
        print("Class balance:")
        for split, pct in pos_percent_info.items():
            print(f"  {split}: {pct:.4f} positive examples")

        
        sample_batch = next(iter(test_dataloader))
        num_input_features = sample_batch[0].shape[-1]  
        num_pert_features = sample_batch[1].shape[-1]   
        num_output_features = sample_batch[2].shape[-1] 
        num_protein = sample_batch[0].shape[1]          
        num_drug_feats = sample_batch[4].shape[1]       

        print("Model input dimensions:")
        print(f"  Protein features: {num_input_features}")
        print(f"  Perturbation features: {num_pert_features}")
        print(f"  Output features: {num_output_features}")
        print(f"  Protein count: {num_protein}")
        print(f"  Drug features: {num_drug_feats}")

        
        print("Initializing model...")
        model = ppODE(
            node_feats=num_input_features,
            pert_feats=num_pert_features,
            hidden_feats=args.hidden_size,
            out_feats=num_output_features,
            pro_feats=num_protein,
            drug_feature_feats=num_drug_feats,
            dropout=args.dropout_rate
        ).to(device)

        
        print(f"Loading model from checkpoint: {args.cp_save_dir_best}")
        checkpoint = torch.load(args.cp_save_dir_best, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        
        print("Making predictions...")
        all_predictions = []
        all_labels = []
        all_proteomics_outputs = []
        all_proteomics_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
                x, pert, y, pheno, fp_phA, fp_phB = [b.to(device) for b in batch]
                outputs, pheno_pred, _ = model(x, pert, fp_phA, fp_phB, args.time_stamp_predict_drug)

                all_predictions.extend(pheno_pred.cpu().numpy())
                all_labels.extend(pheno.cpu().numpy())
                all_proteomics_outputs.extend(outputs.cpu().numpy())
                all_proteomics_labels.extend(y.cpu().numpy())

        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_proteomics_outputs = np.array(all_proteomics_outputs)
        all_proteomics_labels = np.array(all_proteomics_labels)
        binary_predictions = (all_predictions >= 0.5).astype(int)
        experiment_types_for_samples = np.asarray(
            test_dataloader.dataset.get_experiment_types()
        )

        output_lengths = {
            'predicted_probabilities': len(all_predictions),
            'binary_predictions': len(binary_predictions),
            'ground_truth': len(all_labels),
            'proteomics_predictions': len(all_proteomics_outputs),
            'proteomics_ground_truth': len(all_proteomics_labels),
            'experiment_types': len(experiment_types_for_samples),
        }
        if len(set(output_lengths.values())) != 1:
            raise RuntimeError(f"Inconsistent prediction output lengths: {output_lengths}")

        
        results_path = os.path.join(args.dir_save, "predictions.npz")
        np.savez(
            results_path,
            predicted_probabilities=all_predictions,
            binary_predictions=binary_predictions,
            ground_truth=all_labels,
            experiment_types=experiment_types_for_samples
        )

        
        results_path_proteomics = os.path.join(args.dir_save, "predictions_proteomics.npz")
        np.savez(
            results_path_proteomics,
            predicted_probabilities=all_proteomics_outputs,
            ground_truth=all_proteomics_labels,
            experiment_types=experiment_types_for_samples
        )

       
        results_df_path = os.path.join(args.dir_save, "predictions.csv")
        pd.DataFrame({
            'ground_truth': all_labels,
            'predicted_probabilities': all_predictions,
            'binary_predictions': binary_predictions,
            'experiment_type': experiment_types_for_samples
        }).to_csv(results_df_path, index=False)

        print("Predictions completed. Results saved to:")
        print(f"  NPZ format: {results_path}")
        print(f"  CSV format: {results_df_path}")
        print(f"  Total samples: {len(all_predictions)}")
        print(f"  Predicted positive rate: {np.mean(binary_predictions):.4f}")

        return 0

    
    print("Loading and preparing data...")
    train_dataloader, validation_dataloader, test_dataloader, pos_percent_info = prepare_data(args)
    print("Class balance:")
    for split, pct in pos_percent_info.items():
        print(f"  {split}: {pct:.4f} positive examples")

   
    sample_batch = next(iter(train_dataloader))
    num_input_features = sample_batch[0].shape[-1] 
    num_pert_features = sample_batch[1].shape[-1]   
    num_output_features = sample_batch[2].shape[-1]
    num_protein = sample_batch[0].shape[1]         
    num_drug_feats = sample_batch[4].shape[1]      

    print("Model input dimensions:")
    print(f"  Protein features: {num_input_features}")
    print(f"  Perturbation features: {num_pert_features}")
    print(f"  Output features: {num_output_features}")
    print(f"  Protein count: {num_protein}")
    print(f"  Drug features: {num_drug_feats}")

    
    print("Initializing model...")
    model = ppODE(
        node_feats=num_input_features,
        pert_feats=num_pert_features,
        hidden_feats=args.hidden_size,
        out_feats=num_output_features,
        pro_feats=num_protein,
        drug_feature_feats=num_drug_feats,
        dropout=args.dropout_rate
    ).to(device)

    
    optimizer_kwargs = {
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay,
    }
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            momentum=0.95,
            **optimizer_kwargs
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=100,
        verbose=True
    )

    
    if not args.from_scratch and args.cp_save_dir_best:
        print(f"Loading model from checkpoint: {args.cp_save_dir_best}")
        model, optimizer, scheduler, epoch = load_model_from_checkpoint(
            model, optimizer, scheduler, args.cp_save_dir_best, device
        )
        print(f"Continuing from epoch {epoch}")
    else:
        print("Training from scratch")

   
    trainer = Trainer(model, optimizer, scheduler, args, device)

   
    best_val_metrics = trainer.train(train_dataloader, validation_dataloader, test_dataloader)
    with open(os.path.join(args.dir_save, "training_log.pkl"), 'wb') as f:
        pickle.dump(best_val_metrics, f)

    print(f"Training completed. Results saved to {args.dir_save}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
