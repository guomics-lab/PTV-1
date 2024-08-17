import os
import torch
import numpy as np
import pandas as pd
import json
import ast

def str_to_list(s):
    if s is not None and not pd.isna(s):
        s = s.replace("nan", "None")  # change "nan" to "None"
        lst = ast.literal_eval(s)  # convert string to list
        return [x if x is not None else np.nan for x in lst]  # change "None" to np.nan
    else:
        return s

class PerformanceContainer(object):
    """ Simple data class for metrics logging."""
    def __init__(self, data:dict):
        self.data = data

    @staticmethod
    def deep_update(x, y):
        for key in y.keys():
            x.update({key: list(x[key] + y[key])})
        return x

    def to_json(self):
        return json.dumps(self.data)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(data)

# def accuracy(y_hat:torch.Tensor, y:torch.Tensor, threshold: float = 0.1):
#     return torch.mean(((y_hat - y).abs() < threshold).float())

def cor(y_hat: torch.Tensor, y: torch.Tensor):
    # Create a mask for non-NaN elements in both y_hat and y
    mask = (~torch.isnan(y_hat)) & (~torch.isnan(y))

    # Filter y_hat and y using the mask
    y_hat_filtered = y_hat[mask]
    y_filtered = y[mask]

    # Compute the mean of filtered y_hat and y
    y_hat_mean = torch.mean(y_hat_filtered)
    y_mean = torch.mean(y_filtered)

    # Compute the Pearson correlation for filtered y_hat and y
    numerator = torch.sum((y_hat_filtered - y_hat_mean) * (y_filtered - y_mean))
    denominator = torch.sqrt(torch.sum((y_hat_filtered - y_hat_mean)**2) * torch.sum((y_filtered - y_mean)**2))
    
    return numerator / denominator

def pheno_accuracy(probs, labels, pheno_accuracy_threshold, threshold=0.00):    
    # Convert probabilities to binary predictions
    preds = (probs >= threshold).float()
    # Compute accuracy
    # correct = (preds == labels).float().sum()
    correct = ((preds - labels).abs()<=pheno_accuracy_threshold).sum()
    acc = correct / labels.size(0)
    return acc