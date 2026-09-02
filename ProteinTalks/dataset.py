import os
import re
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def str_to_list(s):
    """
    Convert string representation of a list to an actual list
    """
    if s is not None and not pd.isna(s):
        s = s.replace("nan", "None")  # Replace "np.nan" with "None"
        lst = ast.literal_eval(s)  # Convert string to list
        return [x if x is not None else np.nan for x in lst]  # Replace None back to np.nan
    else:
        return s

class ProteomicsDataset(Dataset):
    """
    Dataset class for proteomics data with time series and drug features
    """
    def __init__(self, cellline, data_dir, check_time_point=48):
        """
        Initialize the dataset

        Args:
            cellline: String prefix for the dataset files
            data_dir: Directory containing the dataset files
            check_time_point: Time point to check data validity
        """
        loo_label_name = True

        # Load data files
        self.nodes = pd.read_csv(os.path.join(data_dir, cellline + "node_Index.csv"), header=None)
        expr = pd.read_csv(os.path.join(data_dir, cellline + "expr.csv"), header=None)
        drug_fp_phychemA = pd.read_csv(os.path.join(data_dir, cellline + "drug_fp_phychem_A.csv"), header=None)
        drug_fp_phychemB = pd.read_csv(os.path.join(data_dir, cellline + "drug_fp_phychem_B.csv"), header=None)
        drug_fp_phychemA = drug_fp_phychemA.applymap(str_to_list)
        drug_fp_phychemB = drug_fp_phychemB.applymap(str_to_list)

        if loo_label_name:
            loo_label = pd.read_csv(os.path.join(data_dir, cellline + "loo_label.csv"), header=None)
            timepoint_x, timepoint_y1, timepoint_y2, timepoint_y3 = "0", "6", "24", "48"

        pert = pd.read_csv(os.path.join(data_dir, cellline + "pert.csv"), header=None)
        pheno = pd.read_csv(os.path.join(data_dir, cellline + "pheno.csv"), header=None)
        self.celllines = []
        self.x_data = []
        self.y_data = []
        self.pert_data = []
        self.pheno = []
        self.xdruga = []
        self.xdrugb = []
        self.experiment_types = []

        # Process all experiment types
        all_experiment_type_redu = np.sort(list(set(loo_label[0])))
        all_experiment_type = [i for i in all_experiment_type_redu if "#" in i]  # Only select valid experiments

        for experiment_type in all_experiment_type:
            # Check if data exists for all required time points
            has_all_timepoints = (
                len(loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == 6)]) != 0 and
                len(loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == 24)]) != 0 and
                len(loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == 48)]) != 0
            )

            if has_all_timepoints:
                pattern = re.escape(experiment_type.split('_')[0])  # Only cellline
                self.celllines.append(pattern)

                # Check if data exists at the check time point
                if check_time_point == 0:
                    test_command = loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == int(check_time_point))].index[0]
                else:
                    test_command = loo_label[(loo_label[0].str.contains(pattern)) & (loo_label[1] == int(0))].index[0]

                if (expr.loc[test_command]).any():
                    # Get x data (baseline expression)
                    timepoint = timepoint_x
                    pattern = re.escape(experiment_type.split('_')[0])
                    x_values = expr.loc[loo_label[(loo_label[0].str.contains(pattern)) & (loo_label[1] == int(timepoint))].index[0]].values[0:]
                    x = torch.tensor(np.log(x_values+1)).float().unsqueeze(1)
                    x_min, x_max = torch.min(x), torch.max(x)
                    x_norm = (x - x_min) / (x_max - x_min)
                    self.x_data.append(x_norm)

                    # Get y data (expression at different time points)
                    y_norm_6_24_48 = []
                    for timepoint_y in [timepoint_y1, timepoint_y2, timepoint_y3]:
                        timepoint = timepoint_y
                        y_values = expr.loc[loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == int(timepoint))].index[0]].values[0:]
                        y = torch.tensor(np.log(y_values+1)).float().unsqueeze(1)
                        y_min, y_max = torch.min(y), torch.max(y)
                        y_norm = (y - y_min) / (y_max - y_min)
                        y_norm_6_24_48.append(y_norm)
                    self.y_data.append(torch.stack(y_norm_6_24_48, dim=0))

                    # Get drug features
                    xdruga = drug_fp_phychemA.loc[loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == int(timepoint))].index[0]].values[0]
                    xdrugb = drug_fp_phychemB.loc[loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == int(timepoint))].index[0]].values[0]

                    try:
                        xdruga = torch.tensor(xdruga).float().unsqueeze(1)
                        xdrugb = torch.tensor(xdrugb).float().unsqueeze(1)
                    except:
                        print(f"Error with experiment_type: {experiment_type}")
                        continue

                    # Normalize drug features
                    xdruga_min, xdruga_max = torch.min(xdruga), torch.max(xdruga)
                    xdruga_norm = (xdruga - xdruga_min) / (xdruga_max - xdruga_min)
                    xdrugb_min, xdrugb_max = torch.min(xdrugb), torch.max(xdrugb)
                    xdrugb_norm = (xdrugb - xdrugb_min) / (xdrugb_max - xdrugb_min)
                    self.xdruga.append(xdruga_norm)
                    self.xdrugb.append(xdrugb_norm)

                    # Get perturbation data
                    pert_values = pert.loc[loo_label[loo_label[0] == experiment_type].index[0]].values[0:]
                    pert_tensor = torch.tensor(pert_values).float().unsqueeze(1)
                    self.pert_data.append(pert_tensor)

                    # Get drug-efficacy or combination-synergy labels
                    pheno_values = float(pheno.loc[loo_label[loo_label[0] == experiment_type].index[0]].values)
                    if np.isnan(pheno_values):
                        pheno_tensor = torch.tensor(0.0).float()
                    else:
                        pheno_tensor = torch.tensor(pheno_values).float()
                    self.pheno.append(pheno_tensor)
                    self.experiment_types.append(experiment_type)

    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        return self.x_data[idx], self.pert_data[idx], self.y_data[idx], self.pheno[idx], self.xdruga[idx], self.xdrugb[idx]

    def __len__(self):
        """Return the dataset size"""
        return len(self.x_data)

    def get_experiment_types(self):
        """Return experiment identifiers in dataset order"""
        return list(self.experiment_types)


def clean_dataset(dataset):
    """
    Check dataset for NaN values and replace them with small values

    Args:
        dataset: The ProteomicsDataset to clean

    Returns:
        dataset: The cleaned dataset
        nan_samples: List of sample indices that contained NaN values
    """
    nan_samples = []
    for idx, sample in enumerate(dataset):
        contains_nan = False
        for tensor in sample:
            if torch.isnan(tensor).any():
                contains_nan = True
                tensor[torch.isnan(tensor)] = torch.tensor(1e-6, dtype=torch.float32)  # Replace NaN with small value
        if contains_nan:
            nan_samples.append(idx)

    return dataset, nan_samples

def prepare_data(args):
    """
    Prepare datasets for training, validation, and testing

    Args:
        args: Command line arguments

    Returns:
        train_dataloader: DataLoader for training
        validation_dataloader: DataLoader for validation
        test_dataloader: DataLoader for testing (optional)
        pos_percent_info: Dictionary containing class imbalance information
    """
    dataset = ProteomicsDataset(args.trainval_file_prefix, args.dataset_file_dir, args.check_time_point)
    dataset, nan_samples = clean_dataset(dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Samples with NaN: {len(nan_samples)}")

    # Create test dataset if separate file is provided
    test_dataloader = None
    if args.test_file_prefix != "":
        test_dataset = ProteomicsDataset(args.test_file_prefix, args.dataset_file_dir, args.check_time_point)
        test_dataset, test_nan_samples = clean_dataset(test_dataset)
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Test samples with NaN: {len(test_nan_samples)}")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        all_pheno_test = torch.tensor([i[3] for i in test_dataset])
        pos_percent_test = torch.mean(all_pheno_test).item()
        print(f"Positive percent test: {pos_percent_test:.4f}")

    # Split dataset into train, validation, and test sets
    if args.test_percent < 1e-6:
        train_size = int(args.train_percent * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Save dataset indices
        torch.save(train_dataset.indices, os.path.join(args.dir_save, 'train_indices.pt'))
        torch.save(val_dataset.indices, os.path.join(args.dir_save, 'val_indices.pt'))
    else:
        train_size = int(args.train_percent * len(dataset))
        validation_size = int(args.val_percent * len(dataset))
        test_size = len(dataset) - train_size - validation_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

        # Save dataset indices
        torch.save(train_dataset.indices, os.path.join(args.dir_save, 'train_indices.pt'))
        torch.save(val_dataset.indices, os.path.join(args.dir_save, 'val_indices.pt'))
        torch.save(test_dataset.indices, os.path.join(args.dir_save, 'test_indices.pt'))

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        all_pheno_test = torch.tensor([i[3] for i in test_dataset])
        pos_percent_test = torch.mean(all_pheno_test).item()

    # Calculate class imbalance
    all_pheno_train = torch.tensor([i[3] for i in train_dataset])
    pos_percent_train = torch.mean(all_pheno_train).item()
    print(f"Positive percent train: {pos_percent_train:.4f}")

    all_pheno_val = torch.tensor([i[3] for i in val_dataset])
    pos_percent_val = torch.mean(all_pheno_val).item()
    print(f"Positive percent val: {pos_percent_val:.4f}")

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Collect class balance information
    pos_percent_info = {
        'train': pos_percent_train,
        'val': pos_percent_val
    }

    if args.test_percent > 1e-6 or args.test_file_prefix != "":
        pos_percent_info['test'] = pos_percent_test

    return train_dataloader, validation_dataloader, test_dataloader, pos_percent_info


def prepare_testdata(args):
    """
    Prepare datasets for testing

    Args:
        args: Command line arguments

    Returns:
        test_dataloader: DataLoader for testing (optional)
        pos_percent_info: Dictionary containing class imbalance information
    """

    # Create test dataset
    assert args.test_file_prefix != "", "provide test file"
    test_dataloader = None
    if args.test_file_prefix != "":
        test_dataset = ProteomicsDataset(args.test_file_prefix, args.dataset_file_dir, args.check_time_point)
        test_dataset, test_nan_samples = clean_dataset(test_dataset)
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Test samples with NaN: {len(test_nan_samples)}")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        all_pheno_test = torch.tensor([i[3] for i in test_dataset])
        pos_percent_test = torch.mean(all_pheno_test).item()
        print(f"Positive percent test: {pos_percent_test:.4f}")


    pos_percent_info = {}
    pos_percent_info['test'] = pos_percent_test

    return test_dataloader, pos_percent_info
