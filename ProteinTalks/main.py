"""
This script trains a neural network model for proteomics data using the ppODE architecture.
Args:
    time_stamp_predict_drug (str): Time stamp used to predict drug energy. Options: "6", "24", "48", "all"/"6_24_48".
    lambda_pheno (float): Lambda_pheno for multitask learning.
    taskname_prefix (str): Task name as prefix of checkpoint.
    dataset_file_dir (str): Data directory for the dataset.
    trainval_file_prefix (str): Train validation dataset prefix.
    test_file_prefix (str): If empty, then use test_percent of trainval_file_prefix as test set; else, use test_file_prefix as test set.
    total_epoch (int): Total epoch for training.
    patience (int): Patience for early stopping.
    train_percent (float): Train percent of the dataset.
    val_percent (float): Validation percent of the dataset.
    test_percent (float): Test percent of the dataset. If 0.N, then use test_percent of trainval_file_prefix as test set; else if == 0, use test_file_prefix as test set.
    cp_save_dir_best (str): If empty, the code would run with the best checkpoint by training dataset. Else, if the loss is small enough, the code would save the checkpoint in the cp_save_dir_best.
    batch_size (int): Batch size of running.
Returns:
    Write the checkpoint_test_metric.txt file with the result metric.
"""

import os
import re
import sys

import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import json

from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn.utils import *
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import datetime
import hashlib

import torch.optim as optim
from torch.utils.data.dataset import random_split
from torcheval.metrics.functional import binary_auprc, binary_auroc
from torch.nn.functional import cosine_similarity
from torch import autograd
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import argparse

from utils import *
from model import *
from dataset import *
from plot import *
from mtl import *

parser = argparse.ArgumentParser()
parser.add_argument("--time_stamp_predict_drug", type=str, default="6", help="time stamp used to predict drug senergy, option: 6, 24, 48, all/6_24_48") # ["6", "24", "48", "all"/"6_24_48"]
parser.add_argument("--lambda_pheno", type=float, default=0.8, help="lambda_pheno for multitask learning") #lambda_pheno for multitask learning
parser.add_argument("--taskname_prefix", type=str, default="", help="task name as prefix of checkpoint") # eg multitask_druginfo_
parser.add_argument("--dataset_file_dir", type=str, default="./complete_data_proteo_structured_withcontrol20_0925_allproteins/", help="data dir for the dataset") #data dir for the dataset
parser.add_argument("--trainval_file_prefix", type=str, default="allcelltype_drugpair_", help="train val dataset prefix") #allcelltype_drugpair_crossdrug_
parser.add_argument("--test_file_prefix", type=str, default="", help="if none, then use test_percent of trainval_file_prefix as test set; else, use test_file_prefix as test set") #if "", then use test_percent of trainval_file_prefix as test set; else, use test_file_prefix as test set
parser.add_argument("--total_epoch", type=int, default=5000, help="total epoch for training") #total epoch for training
parser.add_argument("--patience", type=int, default=500, help="patience for early stopping") #patience for early stopping

parser.add_argument("--train_percent", type=float, default = 0.7, help = "train percent of the dataset") #train percent of the dataset
parser.add_argument("--val_percent", type=float, default = 0.2, help = "val percent of the dataset") #val percent of the dataset
parser.add_argument("--test_percent", type=float, default = 0.1, help = "test percent of the dataset") # actually = (1-train_percent-val_percent) #if 0.N, then use test_percent of trainval_file_prefix as test set; else if==0, use test_file_prefix as test set

parser.add_argument("--cp_save_dir_best", type=str, default = "", help = "if none, the code would run with the best checkpoint by training dataset, else if the loss small enough, the code would save the checkpoint in the cp_save_dir_best" ) # in condition "", the code would run with the best checkpoint by training dataset, else if the loss small enough, the code would save the checkpoint in the cp_save_dir_best
parser.add_argument("--batch_size", type=int, default = 128, help = "batch size of running" ) # batch_size of running

args = parser.parse_args()
lambda_pheno = args.lambda_pheno
taskname_prefix = args.taskname_prefix
trainval_file_prefix = args.trainval_file_prefix
test_file_prefix = args.test_file_prefix
total_epoch = args.total_epoch
dataset_file_dir = args.dataset_file_dir
patience = args.patience 
batch_size = args.batch_size

train_percent = args.train_percent
val_percent = args.val_percent
test_percent = args.test_percent

time_stamp_predict_drug = args.time_stamp_predict_drug
cp_save_dir_best = args.cp_save_dir_best

np.set_printoptions(threshold=sys.maxsize)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# seed for repeatability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1995)
torch.cuda.manual_seed(1995)
np.random.seed(1995)

hash_value = hashlib.sha256(str(datetime.datetime.now()).encode("utf8"))
dir_save = './checkpoint/' + taskname_prefix + time_stamp_predict_drug + 'h_' + hash_value.hexdigest() +'/'
if not os.path.exists(dir_save):
	os.makedirs(dir_save)
cp_save_dir = os.path.join(dir_save, str(hash_value.hexdigest())+'checkpoint.pth')

print(f"checkpoint would be saved at {dir_save}")


# Load the dataset
dataset = L1000Dataset(trainval_file_prefix, dataset_file_dir)
print("data num", len(dataset))
print("label class", set([ i[3].item() for i in dataset ]))
# change Nan to 0
nan_samples = []
for idx, sample in enumerate(dataset):
    contains_nan = False
    for tensor in sample:
        if torch.isnan(tensor).any():
            contains_nan = True
            tensor[torch.isnan(tensor)] = torch.tensor(1e-6, dtype=torch.float32) 
    if contains_nan:
        nan_samples.append(idx)
# print("samples with nan:", nan_samples)

# Load the test dataset
if test_file_prefix != "":
    test_dataset = L1000Dataset(test_file_prefix, dataset_file_dir)
    print("test data num", len(dataset))
    nan_samples = []
    for idx, sample in enumerate(test_dataset):
        contains_nan = False
        for tensor in sample:
            if torch.isnan(tensor).any():
                contains_nan = True
                tensor[torch.isnan(tensor)] = torch.tensor(1e-6, dtype=torch.float32)
        if contains_nan:
            nan_samples.append(idx)
    # print("test samples with nan:", nan_samples)

# Logger
logger = PerformanceContainer(data={'train_loss':[], 'train_acc':[],
                                   'test_loss':[], 'test_acc':[],
                                   'forward_time':[], 'backward_time':[],
                                   'train_pro_loss':[],'test_pro_loss':[],
                                   'train_pheno_acc':[],'test_pheno_acc':[],
                                   'train_pheno_loss':[],'test_pheno_loss':[],
                                   'train_pheno_auprc':[],'test_pheno_auprc':[],
                                   'train_pheno_auroc':[],'test_pheno_auroc':[],
                                   })

# Hyperparameters
BATCH_SIZE = batch_size
EPOCHS = total_epoch
verbose_step = 200
accuracy_threshold = 0.01
pheno_accuracy_threshold = 0.00

# Data loader
if test_percent == 0 :
    train_size = int(train_percent * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    ### save index of dataset
    # obtain the index of train_dataset test_dataset
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    # save the index
    torch.save(train_indices, os.path.join(dir_save,'train_indices.pt'))
    torch.save(val_indices, os.path.join(dir_save,'val_indices.pt'))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_pheno_train = torch.tensor([i[3] for i in train_dataset])
    print("pos percent train", torch.mean(all_pheno_train))
    all_pheno_val = torch.tensor([i[3] for i in val_dataset])
    print("pos percent val", torch.mean(all_pheno_val))

else:
    train_size = int(train_percent * len(dataset))
    validation_size = int(val_percent * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    ### save index of dataset validation_dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
    # obtain the index for train_dataset and test_dataset
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    # save the index
    torch.save(train_indices, os.path.join(dir_save,'train_indices.pt'))
    torch.save(val_indices, os.path.join(dir_save,'val_indices.pt'))
    torch.save(test_indices, os.path.join(dir_save,'test_indices.pt'))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_pheno_train = torch.tensor([i[3] for i in train_dataset])
    print("pos percent train", torch.mean(all_pheno_train))
    all_pheno_val = torch.tensor([i[3] for i in val_dataset])
    print("pos percent val", torch.mean(all_pheno_val))
    all_pheno_test = torch.tensor([i[3] for i in test_dataset])
    print("pos percent test", torch.mean(all_pheno_test))

if test_file_prefix != "":
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # load the index
# from torch.utils.data import Subset
# train_indices = torch.load(os.path.join(dir_save,'train_indices.pt'))
# val_indices = torch.load(os.path.join(dir_save,'val_indices.pt'))
# # rebuild train_dataset val_dataset with the loaded index
# train_dataset = Subset(dataset, train_indices)
# val_dataset = Subset(dataset, val_indices)
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# validation_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model definition
num_input_features = dataset[0][0].shape[1]
num_pert_features = dataset[0][1].shape[1]
num_output_features = dataset[0][2].shape[-1]
num_protein = dataset[0][0].shape[0]
num_drug_feats = dataset[0][4].shape[0]
model = ppODE(node_feats=num_input_features, pert_feats=num_pert_features, hidden_feats=32, out_feats=num_output_features, pro_feats=num_protein, drug_feature_feats=num_drug_feats, time_stamp_predict_drug=time_stamp_predict_drug).to(DEVICE)

# Loss function and optimizer
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.95)

# Training loop
best_val_loss = float('inf')
es = 0  # for patience
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    train_acc = 0.0
    train_pro_loss = 0.0
    train_pheno_loss = 0.0
    train_pheno_acc = 0.0
    emb_all = []
    ph_all_train = []
    pheno_predict_all_train = []
    # deltamid_all_train = []
    # deltamax_all_train = []
    # # for data_batch_train, deltamids_batch_train in train_dataloader:
    #     deltamid_all_train.append(deltamids_batch_train[0].to(DEVICE))
    #     deltamax_all_train.append(deltamids_batch_train[1].to(DEVICE))
    for x, pert, y, ph, fp_phA, fp_phB in train_dataloader:
        x, pert, y, ph, fp_phA, fp_phB = x.to(DEVICE), pert.to(DEVICE), y.to(DEVICE), ph.to(DEVICE), fp_phA.to(DEVICE), fp_phB.to(DEVICE)
        outputs, pheno_predict, emb = model(x, pert, fp_phA, fp_phB)        
        # ### use weight addition as total loss
        # loss = (1-lambda_pheno) * mse_loss(outputs, y) + lambda_pheno * bce_loss(pheno_predict, ph)

        ### multitask learning
        # calculate loss
        loss_task1 = mse_loss(outputs, y)
        loss_task2 = bce_loss(pheno_predict, ph)

        # calculate gradients
        optimizer.zero_grad()
        grad_task1 = autograd.grad(loss_task1, model.parameters(), retain_graph=True, allow_unused=True)
        grad_task2 = autograd.grad(loss_task2, model.parameters(), allow_unused=True)
        
        # gradient clipping
        clip_value = 1.0
        grad_task1 = grad_clip(grad_task1, clip_value)
        grad_task2 = grad_clip(grad_task2, clip_value)
        
        # gradient similarity
        similarity = compute_cosine_similarity(grad_task1, grad_task2)
        
        # adjust weights based on similarity
        weight_task1, weight_task2 = adjust_weights_based_on_similarity(similarity, [1-lambda_pheno, lambda_pheno], 0.01)
        
        # update loss
        loss = weight_task1 * loss_task1 + weight_task2 * loss_task2

        # update gradients
        for param, g1, g2 in zip(model.parameters(), grad_task1, grad_task2):
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            if g1 is not None and g2 is not None:
                # check if the gradient size is the same
                if g1.size() == g2.size():
                    param.grad += (g1 + g2)
                else:
                    raise RuntimeError("Gradient size mismatch")
            elif g1 is not None:
                param.grad += g1
            elif g2 is not None:
                param.grad += g2

        # update weights
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        emb_all.append(emb.squeeze(2))
        ph_all_train.append(ph)
        pheno_predict_all_train.append(pheno_predict)
        
        # train_acc = accuracy(y, outputs, accuracy_threshold).item()
        train_acc += cor(y, outputs).item()
        train_pro_loss += mse_loss(outputs, y).item()
        train_pheno_loss += bce_loss(pheno_predict, ph).item()
        train_pheno_acc += pheno_accuracy(pheno_predict, ph, pheno_accuracy_threshold).item()

    auprc_train = binary_auprc(torch.cat(pheno_predict_all_train), torch.cat(ph_all_train)).item() #binary_auroc(input, target)
    auroc_train = binary_auroc(torch.cat(pheno_predict_all_train), torch.cat(ph_all_train)).item()
    
    # validation
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_pro_loss = 0.0
        test_pheno_loss = 0.0
        test_pheno_acc = 0.0
        outputs_all = []
        y_all = []
        ph_all_test = []
        pheno_predict_all_test = []
        deltamid_all_test = []
        deltamax_all_test = []
        # for data_batch_test, deltamids_batch_test in test_dataloader:
        #     deltamid_all_test.append(deltamids_batch_test[0].to(DEVICE))
        #     deltamax_all_test.append(deltamids_batch_test[1].to(DEVICE))
        for x, pert, y, ph, fp_phA, fp_phB in validation_dataloader:
            x, pert, y, ph, fp_phA, fp_phB = x.to(DEVICE), pert.to(DEVICE), y.to(DEVICE), ph.to(DEVICE), fp_phA.to(DEVICE), fp_phB.to(DEVICE)
            outputs, pheno_predict, emb = model(x, pert, fp_phA, fp_phB)
            loss = (1-lambda_pheno)*mse_loss(outputs, y) + lambda_pheno*bce_loss(pheno_predict, ph)
            test_loss += loss.item()
            emb_all.append(emb.squeeze(2))
            outputs_all.append(outputs)
            y_all.append(y)
            ph_all_test.append(ph)
            pheno_predict_all_test.append(pheno_predict)
            # test_acc = accuracy(y_all, outputs_all, accuracy_threshold).item()
            test_acc += cor(y, outputs).item()
            test_pro_loss += mse_loss(outputs, y).item()
            test_pheno_loss += bce_loss(pheno_predict, ph).item()
            test_pheno_acc += pheno_accuracy(pheno_predict, ph, pheno_accuracy_threshold).item()
            
        auprc_test = binary_auprc(torch.cat(pheno_predict_all_test), torch.cat(ph_all_test)).item()
        auroc_test = binary_auroc(torch.cat(pheno_predict_all_test), torch.cat(ph_all_test)).item()
        
        logger.deep_update(logger.data, dict(train_loss=[total_loss / len(train_dataloader)], train_acc=[train_acc / len(train_dataloader)],
                                             test_loss=[test_loss / len(validation_dataloader)], test_acc=[test_acc / len(validation_dataloader)],
                                             train_pro_loss = [train_pro_loss / len(train_dataloader)], test_pro_loss = [test_pro_loss / len(validation_dataloader)],
                                             train_pheno_loss = [train_pheno_loss / len(train_dataloader)], test_pheno_loss = [test_pheno_loss / len(validation_dataloader)],
                                             train_pheno_acc = [train_pheno_acc / len(train_dataloader)],  test_pheno_acc = [test_pheno_acc / len(validation_dataloader)],                              
                                             train_pheno_auprc = [auprc_train], test_pheno_auprc = [auprc_test],
                                             train_pheno_auroc = [auroc_train], test_pheno_auroc = [auroc_test],
                                            )
        )

        
    if epoch % verbose_step == 0: 
        print('[{}], Train Loss: {:3.3f}, Test loss: {:3.3f}, Train Accuracy: {:3.3f}, Test Accuracy: {:3.3f}, Train pro loss: {:3.3f}, Test pro loss: {:3.3f}, Train pheno loss: {:3.3f}, Test pheno loss: {:3.3f}, Train pheno acc: {:3.3f}, Test pheno acc: {:3.3f}, Train pheno auprc: {:3.3f}, Test pheno auprc: {:3.3f}, Train pheno auroc: {:3.3f}, Test pheno auroc: {:3.3f} '.format(epoch,
                                                                                            total_loss / len(train_dataloader),
                                                                                            test_loss / len(validation_dataloader),
                                                                                            train_acc / len(train_dataloader),
                                                                                            test_acc / len(validation_dataloader),
                                                                                            train_pro_loss/ len(train_dataloader), test_pro_loss/ len(validation_dataloader),
                                                                                            train_pheno_loss/ len(train_dataloader), test_pheno_loss/ len(validation_dataloader),
                                                                                            train_pheno_acc/ len(train_dataloader), test_pheno_acc/ len(validation_dataloader),
                                                                                            auprc_train,  auprc_test,
                                                                                            auroc_train,  auroc_test,
                                                                                            ))
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        cp_save_dir = dir_save+str(epoch)+'_checkpoint.pth'
        torch.save(checkpoint, cp_save_dir)
    
    if test_pheno_loss < best_val_loss:
        best_val_loss = test_pheno_loss
        es = 0
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        cp_save_dir = dir_save+str(epoch)+'best_checkpoint.pth'
        cp_save_dir_best = cp_save_dir # the test will be run in the cp_save_dir
        torch.save(checkpoint, cp_save_dir)
    else:
        es += 1
        if es > patience:
            print("Early stopping with best_val_loss: ", best_val_loss, "and val_loss for this epoch: ", test_pheno_loss)
            break

print('[{}], Train Loss: {:3.3f}, Test loss: {:3.3f}, Train Accuracy: {:3.3f}, Test Accuracy: {:3.3f}, Train pro loss: {:3.3f}, Test pro loss: {:3.3f}, Train pheno loss: {:3.3f}, Test pheno loss: {:3.3f}, Train pheno acc: {:3.3f}, Test pheno acc: {:3.3f}, Train pheno auprc: {:3.3f}, Test pheno auprc: {:3.3f}, Train pheno auroc: {:3.3f}, Test pheno auroc: {:3.3f} '.format(epoch,
                                                                                    total_loss / len(train_dataloader),
                                                                                    test_loss / len(validation_dataloader),
                                                                                    train_acc / len(train_dataloader),
                                                                                    test_acc / len(validation_dataloader),
                                                                                    train_pro_loss/ len(train_dataloader), test_pro_loss/ len(validation_dataloader),
                                                                                    train_pheno_loss/ len(train_dataloader), test_pheno_loss/ len(validation_dataloader),
                                                                                    train_pheno_acc/ len(train_dataloader), test_pheno_acc/ len(validation_dataloader),
                                                                                    auprc_train,  auprc_test,
                                                                                    auroc_train,  auroc_test,
                                                                                    ))

print("predict pos percent train", torch.mean(torch.cat(pheno_predict_all_train)))
print("predict pos percent test", torch.mean(torch.cat(pheno_predict_all_test)))

with open(dir_save+"checkpoint_test_metric.txt", "w") as f:
    f.write("train result ...\n")
    f.write('[{}], Train Loss: {:3.3f}, Test loss: {:3.3f}, Train Accuracy: {:3.3f}, Test Accuracy: {:3.3f}, Train pro loss: {:3.3f}, Test pro loss: {:3.3f}, Train pheno loss: {:3.3f}, Test pheno loss: {:3.3f}, Train pheno acc: {:3.3f}, Test pheno acc: {:3.3f}, Train pheno auprc: {:3.3f}, Test pheno auprc: {:3.3f}, Train pheno auroc: {:3.3f}, Test pheno auroc: {:3.3f} '.format(epoch,
                                                                                    total_loss / len(train_dataloader),
                                                                                    test_loss / len(validation_dataloader),
                                                                                    train_acc / len(train_dataloader),
                                                                                    test_acc / len(validation_dataloader),
                                                                                    train_pro_loss/ len(train_dataloader), test_pro_loss/ len(validation_dataloader),
                                                                                    train_pheno_loss/ len(train_dataloader), test_pheno_loss/ len(validation_dataloader),
                                                                                    train_pheno_acc/ len(train_dataloader), test_pheno_acc/ len(validation_dataloader),
                                                                                    auprc_train,  auprc_test,
                                                                                    auroc_train,  auroc_test,
                                                                                    ))

    f.write("predict pos percent train %s"%(torch.mean(torch.cat(pheno_predict_all_train))))
    f.write('\n')
    f.write("predict pos percent test %s"%(torch.mean(torch.cat(pheno_predict_all_test))))
    f.write('\n')
    f.write("\npredict pos percent train %s"%( torch.mean(torch.cat(pheno_predict_all_train))))
    f.write("\npredict pos percent test %s"%( torch.mean(torch.cat(pheno_predict_all_test))))


### save validation value ...
with torch.no_grad():
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_pro_loss = 0.0
    test_pheno_loss = 0.0
    test_pheno_acc = 0.0
    # test_pheno_auprc = 0.0
    # test_pheno_auroc = 0.0        
    outputs_all = []
    y_all = []
    ph_all_test = []
    pheno_predict_all_test = []
    deltamid_all_test = []
    deltamax_all_test = []
    # for data_batch_test, deltamids_batch_test in test_dataloader:
    #     deltamid_all_test.append(deltamids_batch_test[0].to(DEVICE))
    #     deltamax_all_test.append(deltamids_batch_test[1].to(DEVICE))
    for x, pert, y, ph, fp_phA, fp_phB in validation_dataloader:
        x, pert, y, ph, fp_phA, fp_phB = x.to(DEVICE), pert.to(DEVICE), y.to(DEVICE), ph.to(DEVICE), fp_phA.to(DEVICE), fp_phB.to(DEVICE)
        outputs, pheno_predict, emb = model(x, pert, fp_phA, fp_phB)
        loss = (1-lambda_pheno)*mse_loss(outputs, y) + lambda_pheno*bce_loss(pheno_predict, ph)
        test_loss += loss.item()
        emb_all.append(emb.squeeze(2))
        outputs_all.append(outputs)
        y_all.append(y)
        ph_all_test.append(ph)
        pheno_predict_all_test.append(pheno_predict)
        # test_acc = accuracy(y_all, outputs_all, accuracy_threshold).item()
        test_acc += cor(y, outputs).item()
        test_pro_loss += mse_loss(outputs, y).item()
        test_pheno_loss += bce_loss(pheno_predict, ph).item()
        test_pheno_acc += pheno_accuracy(pheno_predict, ph, pheno_accuracy_threshold).item()

    auprc_test = binary_auprc(torch.cat(pheno_predict_all_test), torch.cat(ph_all_test)).item() #input, target
    auroc_test = binary_auroc(torch.cat(pheno_predict_all_test), torch.cat(ph_all_test)).item()
    
    pheno_predict_np = torch.cat(pheno_predict_all_test).cpu().numpy()
    ph_all_np = torch.cat(ph_all_test).cpu().numpy()

    threshold = 0.5
    pheno_predict_binary = (pheno_predict_np >= threshold).astype(int)

    # cal Precision, Recall, F1-score, Matthew's Correlation Coefficient, Cohen's Kappa
    accuracy = accuracy_score(ph_all_np, pheno_predict_binary)
    precision = precision_score(ph_all_np, pheno_predict_binary)
    recall = recall_score(ph_all_np, pheno_predict_binary)
    f1 = f1_score(ph_all_np, pheno_predict_binary)
    mcc = matthews_corrcoef(ph_all_np, pheno_predict_binary)
    kappa = cohen_kappa_score(ph_all_np, pheno_predict_binary)

    # print metric
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)
    print("Matthew's Correlation Coefficient: ", mcc)
    print("Cohen's Kappa: ", kappa)
    print('[{}], Train Loss: {:3.3f}, Train Accuracy: {:3.3f},  Train pro loss: {:3.3f}, Train pheno loss: {:3.3f},  Train pheno acc: {:3.3f},  Train pheno auprc: {:3.3f}, Train pheno auroc: {:3.3f} '.format(epoch,
                                                                                    total_loss / len(train_dataloader),
                                                                                    train_acc / len(train_dataloader),
                                                                                    train_pro_loss/ len(train_dataloader),
                                                                                    train_pheno_loss/ len(train_dataloader),
                                                                                    train_pheno_acc/ len(train_dataloader),
                                                                                    auprc_train, 
                                                                                    auroc_train, 
                                                                                    ))
    print('[{}], Test loss: {:3.3f},  Test Accuracy: {:3.3f},  Test pro loss: {:3.3f},  Test pheno loss: {:3.3f},   Test pheno acc: {:3.3f},   Test pheno auprc: {:3.3f},   Test pheno auroc: {:3.3f} '.format(epoch,
                                                                                    test_loss / len(validation_dataloader),
                                                                                    test_acc / len(validation_dataloader),
                                                                                    test_pro_loss/ len(validation_dataloader),
                                                                                    test_pheno_loss/ len(validation_dataloader),
                                                                                    test_pheno_acc/ len(validation_dataloader),
                                                                                    auprc_test,
                                                                                    auroc_test,
                                                                                    ))
     
with open(dir_save+"checkpoint_test_metric.txt", "a+") as f:
    f.write("\n\nvalidation result ...\n")
    f.write("Accuracy: %s"%(accuracy))
    f.write('\n')
    f.write("Precision: %s"%(precision))
    f.write('\n')
    f.write("Recall: %s"%(recall))
    f.write('\n')
    f.write("F1-score: %s"%(f1))
    f.write('\n')
    f.write("Matthew's Correlation Coefficient: %s"%(mcc))
    f.write('\n')
    f.write("Cohen's Kappa: %s"%(kappa))
    f.write('\n')
    f.write('[{}], Train Loss: {:3.3f}, Train Accuracy: {:3.3f},  Train pro loss: {:3.3f}, Train pheno loss: {:3.3f},  Train pheno acc: {:3.3f},  Train pheno auprc: {:3.3f}, Train pheno auroc: {:3.3f} '.format(epoch,
                                                                                    total_loss / len(train_dataloader),
                                                                                    train_acc / len(train_dataloader),
                                                                                    train_pro_loss/ len(train_dataloader),
                                                                                    train_pheno_loss/ len(train_dataloader),
                                                                                    train_pheno_acc/ len(train_dataloader),
                                                                                    auprc_train, 
                                                                                    auroc_train, 
                                                                                    ))
    f.write('\n')
    f.write('[{}], Test loss: {:3.3f},  Test Accuracy: {:3.3f},  Test pro loss: {:3.3f},  Test pheno loss: {:3.3f},   Test pheno acc: {:3.3f},   Test pheno auprc: {:3.3f},   Test pheno auroc: {:3.3f} '.format(epoch,
                                                                                    test_loss / len(validation_dataloader),
                                                                                    test_acc / len(validation_dataloader),
                                                                                    test_pro_loss/ len(validation_dataloader),
                                                                                    test_pheno_loss/ len(validation_dataloader),
                                                                                    test_pheno_acc/ len(validation_dataloader),
                                                                                    auprc_test,
                                                                                    auroc_test,
                                                                                    ))


with open(dir_save+'/logger.json', 'w') as output:
    json_str = logger.to_json()
    output.write(json_str)


#### load the saved checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': EPOCHS
}
print(str(datetime.datetime.now()))
cp_save_dir = dir_save+str(epoch)+'_checkpoint.pth'
torch.save(checkpoint, cp_save_dir)
print(cp_save_dir)

#### Load the model checkpoint
checkpoint = torch.load(cp_save_dir_best) # the test will be run in the cp_save_dir_best
model = ppODE(node_feats=num_input_features, pert_feats=num_pert_features, hidden_feats=32, out_feats=num_output_features, pro_feats=num_protein, drug_feature_feats=num_drug_feats, time_stamp_predict_drug=time_stamp_predict_drug).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.95)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loaded_epoch = checkpoint['epoch']
print("Model and optimizer state loaded successfully, epoch:", loaded_epoch)


### plot the loss and accuracy curve for training and validation
fl_n = 10
plot_loss(logger.data['train_loss'], logger.data['test_loss'], fl_n, dir_save)
plot_pheno(logger.data['train_pheno_loss'], logger.data['train_pheno_acc'], logger.data['test_pheno_loss'], logger.data['test_pheno_acc'], fl_n, dir_save)
plot_auprc_auroc(logger.data['train_pheno_auprc'], logger.data['train_pheno_auroc'], logger.data['test_pheno_auprc'], logger.data['test_pheno_auroc'], fl_n, dir_save)
plot_proteomics_predict(logger.data['train_pro_loss'], logger.data['train_acc'], logger.data['test_pro_loss'], logger.data['test_acc'], fl_n, dir_save)

pheno_predict_np = torch.cat(pheno_predict_all_test).cpu().numpy()
ph_all_np = torch.cat(ph_all_test).cpu().numpy()
plot_auprc(pheno_predict_np, ph_all_np, dir_save, "validation")
plot_auroc(pheno_predict_np, ph_all_np, dir_save, "validation")

### test 
if test_percent != 0 or test_file_prefix!='':
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_pro_loss = 0.0
        test_pheno_loss = 0.0
        test_pheno_acc = 0.0
        outputs_all = []
        y_all = []
        ph_all_test = []
        pheno_predict_all_test = []
        deltamid_all_test = []
        deltamax_all_test = []
        # for data_batch_test, deltamids_batch_test in test_dataloader:
        #     deltamid_all_test.append(deltamids_batch_test[0].to(DEVICE))
        #     deltamax_all_test.append(deltamids_batch_test[1].to(DEVICE))
        for x, pert, y, ph, fp_phA, fp_phB in test_dataloader:
            x, pert, y, ph, fp_phA, fp_phB = x.to(DEVICE), pert.to(DEVICE), y.to(DEVICE), ph.to(DEVICE), fp_phA.to(DEVICE), fp_phB.to(DEVICE)
            outputs, pheno_predict, emb = model(x, pert, fp_phA, fp_phB)
            loss = (1-lambda_pheno)*mse_loss(outputs, y) + lambda_pheno*bce_loss(pheno_predict, ph)
            test_loss += loss.item()
            emb_all.append(emb.squeeze(2))
            outputs_all.append(outputs)
            y_all.append(y)
            ph_all_test.append(ph)
            pheno_predict_all_test.append(pheno_predict)
            test_acc += cor(y, outputs).item()
            test_pro_loss += mse_loss(outputs, y).item()
            test_pheno_loss += bce_loss(pheno_predict, ph).item()
            test_pheno_acc += pheno_accuracy(pheno_predict, ph, pheno_accuracy_threshold).item()

        auprc_test = binary_auprc(torch.cat(pheno_predict_all_test), torch.cat(ph_all_test)).item() #input, target
        auroc_test = binary_auroc(torch.cat(pheno_predict_all_test), torch.cat(ph_all_test)).item()
        
        pheno_predict_np = torch.cat(pheno_predict_all_test).cpu().numpy()
        ph_all_np = torch.cat(ph_all_test).cpu().numpy()

        threshold = 0.5
        pheno_predict_binary = (pheno_predict_np >= threshold).astype(int)

        # Precision, Recall, F1-score, Matthew's Correlation Coefficient, Cohen's Kappa
        accuracy = accuracy_score(ph_all_np, pheno_predict_binary)
        precision = precision_score(ph_all_np, pheno_predict_binary)
        recall = recall_score(ph_all_np, pheno_predict_binary)
        f1 = f1_score(ph_all_np, pheno_predict_binary)
        mcc = matthews_corrcoef(ph_all_np, pheno_predict_binary)
        kappa = cohen_kappa_score(ph_all_np, pheno_predict_binary)

        # print metric
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1-score: ", f1)
        print("Matthew's Correlation Coefficient: ", mcc)
        print("Cohen's Kappa: ", kappa)
        
        print('[{}], Train Loss: {:3.3f}, Train Accuracy: {:3.3f},  Train pro loss: {:3.3f}, Train pheno loss: {:3.3f},  Train pheno acc: {:3.3f},  Train pheno auprc: {:3.3f}, Train pheno auroc: {:3.3f} '.format(loaded_epoch,
                                                                                        total_loss / len(train_dataloader),
                                                                                        train_acc / len(train_dataloader),
                                                                                        train_pro_loss/ len(train_dataloader),
                                                                                        train_pheno_loss/ len(train_dataloader),
                                                                                        train_pheno_acc/ len(train_dataloader),
                                                                                        auprc_train, 
                                                                                        auroc_train, 
                                                                                        ))
        print('[{}], Test loss: {:3.3f},  Test Accuracy: {:3.3f},  Test pro loss: {:3.3f},  Test pheno loss: {:3.3f},   Test pheno acc: {:3.3f},   Test pheno auprc: {:3.3f},   Test pheno auroc: {:3.3f} '.format(loaded_epoch,
                                                                                        test_loss / len(test_dataloader),
                                                                                        test_acc / len(test_dataloader),
                                                                                        test_pro_loss/ len(test_dataloader),
                                                                                        test_pheno_loss/ len(test_dataloader),
                                                                                        test_pheno_acc/ len(test_dataloader),
                                                                                        auprc_test,
                                                                                        auroc_test,
                                                                                        ))
    len_test_dataloader = len(test_dataloader)
    # save test metric
    with open(dir_save+"checkpoint_test_metric.txt", "a+") as f:
        f.write("\n\ntest result ...\n")
        f.write("Accuracy: %s"%(accuracy))
        f.write('\n')
        f.write("Precision: %s"%(precision))
        f.write('\n')
        f.write("Recall: %s"%(recall))
        f.write('\n')
        f.write("F1-score: %s"%(f1))
        f.write('\n')
        f.write("Matthew's Correlation Coefficient: %s"%(mcc))
        f.write('\n')
        f.write("Cohen's Kappa: %s"%(kappa))
        f.write('\n')
        f.write('[{}], Train Loss: {:3.3f}, Train Accuracy: {:3.3f},  Train pro loss: {:3.3f}, Train pheno loss: {:3.3f},  Train pheno acc: {:3.3f},  Train pheno auprc: {:3.3f}, Train pheno auroc: {:3.3f} '.format(epoch,
                                                                                        total_loss / len(train_dataloader),
                                                                                        train_acc / len(train_dataloader),
                                                                                        train_pro_loss/ len(train_dataloader),
                                                                                        train_pheno_loss/ len(train_dataloader),
                                                                                        train_pheno_acc/ len(train_dataloader),
                                                                                        auprc_train, 
                                                                                        auroc_train, 
                                                                                        ))
        f.write('\n')
        f.write('[{}], Test loss: {:3.3f},  Test Accuracy: {:3.3f},  Test pro loss: {:3.3f},  Test pheno loss: {:3.3f},   Test pheno acc: {:3.3f},   Test pheno auprc: {:3.3f},   Test pheno auroc: {:3.3f} '.format(epoch,
                                                                                        test_loss / len_test_dataloader,
                                                                                        test_acc / len_test_dataloader,
                                                                                        test_pro_loss/ len_test_dataloader,
                                                                                        test_pheno_loss/ len_test_dataloader,
                                                                                        test_pheno_acc/ len_test_dataloader,
                                                                                        auprc_test,
                                                                                        auroc_test,
                                                                                        ))
        f.write("\npredict pos percent train %s"%(torch.mean(torch.cat(pheno_predict_all_train))))
        f.write("\npredict pos percent test %s"%(torch.mean(torch.cat(pheno_predict_all_test))))

    print("predict pos percent train", torch.mean(torch.cat(pheno_predict_all_train)))
    print("predict pos percent test", torch.mean(torch.cat(pheno_predict_all_test)))

    pheno_predict_np = torch.cat(pheno_predict_all_test).cpu().numpy()
    ph_all_np = torch.cat(ph_all_test).cpu().numpy()

    # plot auroc and auprc
    plot_auroc(pheno_predict_np, ph_all_np, dir_save, "test")
    plot_auprc(pheno_predict_np, ph_all_np, dir_save, "test")

else:
    accuracy, precision, recall, f1, mcc, kappa = "without test dataset", "without test dataset", "without test dataset", "without test dataset", "without test dataset", "without test dataset"
    
print(epoch, dir_save)