import argparse
import os
import torch
import numpy as np
import datetime
import hashlib

def get_args():
    """
    Get command line arguments

    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    # Directory and file settings
    parser.add_argument("--dir_save", type=str, default="./results",
                        help="Base directory for saving results")
    parser.add_argument("--dataset_file_dir", type=str, required=True,
                        help="Directory containing dataset files")
    parser.add_argument("--trainval_file_prefix", type=str, required=False,
                        help="Prefix for train/validation files")
    parser.add_argument("--test_file_prefix", type=str, default="",
                        help="Prefix for test files")
    parser.add_argument("--taskname_prefix", type=str, default="",
                        help="Prefix for task name")
    parser.add_argument("--time_stamp_predict_drug", type=str, default="6_24_48",
                        choices=["6_24_48"],
                        help="Time points used for drug prediction")

    # Training parameters
    parser.add_argument("--total_epoch", type=int, default=1000,
                        help="Total number of epochs")
    parser.add_argument("--patience", type=int, default=500,
                        help="Patience for early stopping")
    parser.add_argument("--train_percent", type=float, default=0.7,
                        help="Percentage of data used for training")
    parser.add_argument("--val_percent", type=float, default=0.2,
                        help="Percentage of data used for validation")
    parser.add_argument("--test_percent", type=float, default=0.1,
                        help="Percentage of data used for testing")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Tolerance for ODE solver")

    # Model architecture parameters
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Hidden layer size for the model")
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                        help="Dropout rate for the model")

    # Optimizer and learning rate parameters
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["sgd", "adam", "adamw"],
                        help="Optimizer to use (sgd, adam, or adamw)")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay rate")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of epochs for learning rate warmup")
    parser.add_argument("--use_grad_optim", action="store_true",
                        help="Use gradient optimization techniques")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm")

    # Checkpoint handling
    parser.add_argument("--cp_save_dir_best", type=str, default="",
                        help="Path to a model checkpoint for prediction")
    parser.add_argument("--train_from_scratch", type=str, default="from_scratch",
                        choices=["from_scratch", "predict"],
                        help="Run training from scratch or prediction")
    parser.add_argument("--check_time_point", type=int, default=48,
                        help="Time point to check whether there is info in the data")

    # Loss calculation parameters
    parser.add_argument("--lambda_pheno", type=float, default=0.8,
                        help="Weight for phenotype prediction loss")

    parser.add_argument("--indices_dir", type=str,
                        help="indices dir to split dataset")
    parser.add_argument("--indices_prefix", type=str, default="",
                        help="indices prefix to split dataset")
    parser.add_argument("--cancer_type", type=str, default=None,
                        help="file save cancer type")

    # SWAG parameters
    parser.add_argument("--use_swag", action="store_true",
                        help="Use SWAG (Stochastic Weight Averaging-Gaussian)")
    parser.add_argument("--swag_lr", type=float, default=0.0005,
                        help="Learning rate for the SWAG phase")
    parser.add_argument("--swag_freq", type=int, default=1,
                        help="Frequency of collecting SWAG snapshots (in epochs)")
    parser.add_argument("--swag_max_models", type=int, default=20,
                        help="Maximum number of models to store for covariance estimation")
    parser.add_argument("--swag_start_factor", type=float, default=0.2,
                        help="LR factor relative to initial LR to trigger SWAG start")
    parser.add_argument("--swag_samples", type=int, default=30,
                        help="Number of SWAG samples for uncertainty estimation")

    parser.add_argument("--random_seed", type=int, default=1995,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Process args and set derived parameters
    if args.train_from_scratch == "from_scratch" and args.cp_save_dir_best == "":
        args.from_scratch = True
    else:
        args.from_scratch = False

    # Ensure test_percent is valid if not specified
    if args.test_percent < 1e-6:
        args.test_percent = 1 - args.train_percent - args.val_percent

    return args

def setup_device(args):
    """
    Set up and return the device (CPU/GPU) to use
    """

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seeds for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    return device

def setup_directories(args):
    """
    Set up directories for saving checkpoints and results
    """

    # Create a unique hash for this run
    hash_value = hashlib.sha256(str(datetime.datetime.now()).encode("utf8"))

    # Create directory for saving checkpoints
    dir_save = os.path.join(args.dir_save,
                           args.taskname_prefix + args.time_stamp_predict_drug + 'h_' + hash_value.hexdigest() + '/')

    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    return dir_save
