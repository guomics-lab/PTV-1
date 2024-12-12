import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.module import Module
from torchdyn.utils import *
from torchdyn.core import NeuralODE

class FullyConnectedLayer(nn.Module):
    def __init__(self, in_feats:int, out_feats:int, activation,
                 dropout:int, bias:bool=True):
        super().__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc.weight.size(1))
        self.fc.weight.data.uniform_(-stdv, stdv)
        if self.fc.bias is not None:
            self.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        h = self.fc(h)
        if self.activation:
            h = self.activation(h)
        return h


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ppODE(nn.Module):
    def __init__(self, node_feats, pert_feats, hidden_feats, out_feats, pro_feats, drug_feature_feats, time_stamp_predict_drug):
        super(ppODE, self).__init__()
        self.mid_feats = 32
        self.mid_feats_drugsens = 32
        self.mid_feats_drugs = 32
        self.mid_feats_drugsens_drugs = 32        
        self.linear_input = FullyConnectedLayer(in_feats=2, out_feats=self.mid_feats, activation=nn.Softplus(), dropout=0)
        
        # Convolution layers
        self.conv1 = nn.Conv1d(self.mid_feats, hidden_feats, kernel_size=1)  # Kernel size is 1 for 1D data
        self.conv2 = nn.Conv1d(hidden_feats, self.mid_feats, kernel_size=1)  # Kernel size is 1 for 1D data
        
        self.convdrug1 = nn.Conv1d(2, hidden_feats, kernel_size=2)  # Kernel size is 1 for 1D data
    
        # NeuralODE
        func = nn.Sequential(
            FullyConnectedLayer(in_feats=hidden_feats, out_feats=hidden_feats, activation=nn.Softplus(), dropout=0.1),
            FullyConnectedLayer(in_feats=hidden_feats, out_feats=hidden_feats, activation=None, dropout=0)
        )

        self.neuralDE = NeuralODE(func, solver='rk4') #single time point 
        
        # Final Fully connected layer
        self.layer_final = nn.Linear(self.mid_feats, out_feats)
        self.time_tick_num = 4
        self.time_stamp_predict_drug = time_stamp_predict_drug
        if self.time_stamp_predict_drug in ['6', '24', '48']:
            self.drugsens_conv1 = nn.Conv1d(pro_feats, self.mid_feats_drugsens, kernel_size=2)
        else: # time point 0+ all time(6, 24, 48)
            self.drugsens_conv1 = nn.Conv1d(pro_feats, self.mid_feats_drugsens, kernel_size=self.time_tick_num)
        self.drugs_conv2 = nn.Conv1d(drug_feature_feats, self.mid_feats_drugs, kernel_size=2)
        self.pheno_fc1 = nn.Linear(self.mid_feats_drugsens + self.mid_feats_drugs, self.mid_feats_drugsens_drugs) 
        self.pheno_fc2 = nn.Linear(self.mid_feats_drugsens_drugs, 1)
        

    def forward(self, x, pert, fp_phA, fp_phB):
        # Extracting features
        emb_xpert = torch.cat([x, pert], dim=-1)  # Shape [batch_size, 426 len feature, 2]
        batch_size = emb_xpert[0]
        
        # Concatenate
        emb_combined = self.linear_input(emb_xpert) # Shape [batch_size, 426len, self.mid_feats]
        emb_combined = torch.transpose(emb_combined, 1, 2)   # change channel dimension: [batch_size, self.mid_feats, 426], for conv1d [N,C,L] N is a batch size, C denotes a number of channels, L is a length of signal sequence.
        
        # Convolution
        emb_cnn = self.conv1(emb_combined) #([batch_size, 128 hidden_feats, 426])
        
        fp_ph_combined = torch.cat([fp_phA, fp_phB], dim=-1)  # Shape [batch_size, 935 featurelen, 2]
        fp_ph_combined = torch.transpose(fp_ph_combined, 1, 2)  # Shape [batch_size, self.mid_feats, 935 featurelen]
        fp_ph_cnn = self.convdrug1(fp_ph_combined)  # Shape [batch_size, 935 featurelen, self.mid_feats]
        
        emb_cnn = torch.transpose(emb_cnn, 1, 2)  #([batch_size, 426, 128 hidden_feats])
        
        # NeuralODE
        time_tick_num = self.time_tick_num
        emb_ode = self.neuralDE(emb_cnn, torch.linspace(0,time_tick_num-1,time_tick_num))  #[torch.Size([time_tick_num]), torch.Size([time_tick_num, batch_size, 426 feature_num, 128])]
        emb_ode = emb_ode[1][1:] #[batch_size, 426, 128] #get the trajectory part[1], and the last time point [-1]   #([time_tick_num-1, batch_size, 469, 128])

        # Second convolution
        emb_ode = torch.transpose(emb_ode, -2, -1) # [batch_size, 128, 426], #([time_tick_num-1, batch_size, 128, 469])
        emb_ode = torch.transpose(emb_ode, 0, 1) #([batch_size, time_tick_num-1, 128, 469]) 
        emb_ode_reshape = emb_ode.reshape(-1, emb_ode.size(-2), emb_ode.size(-1)) # ([batch_size * time_tick_num-1, 128, 469]) 

        emb_ode_conv = self.conv2(emb_ode_reshape) #[batch_size, 24, 426], #[batch_size, 32?, 426] -> #([batch_size * time_tick_num-1, 32, 469])
        emb_ode_conv = torch.transpose(emb_ode_conv, -2, -1) #[batch_size, 426, 24] #[batch_size, 426, 32?] -> #([batch_size * time_tick_num-1, 469, 32])

        emb_ode_reshape_2 = emb_ode_conv.view(emb_ode.size(0), emb_ode.size(1), emb_ode_conv.size(1), emb_ode_conv.size(2)) #([batch_size, time_tick_num-1, 469, 32])
        
        # Final layer
        y = self.layer_final(emb_ode_reshape_2) ##[4, 469, 1]  #([batch_size, time_tick_num-1, 469, 1])  eg y.shape torch.Size([4, 3, 469, 1])
        
        ######## prediction of drug 
        bs = x.shape[0]
        if self.time_stamp_predict_drug == '6':
            xy = torch.cat([x, y[:, 0, :, :].squeeze(1)], dim=-1) #[4, 426, 2]  # time point 0 6h
        elif self.time_stamp_predict_drug == '24':
            xy = torch.cat([x, y[:, 1, :, :].squeeze(1)], dim=-1) #[4, 426, 2]  # time point 1 24h
        elif self.time_stamp_predict_drug == '48':
            xy = torch.cat([x, y[:, 2, :, :].squeeze(1)], dim=-1) #[4, 426, 2]  # time point 2 48h
        else: ## time point 0+ all time(6, 24, 48)  
            xy = torch.cat([x, torch.transpose(y, -3, -1).squeeze(1)], dim=-1) # y [4, 3, 469, 1] -> [4, 1, 469, 3] -> [4, 469, 3]; later +x: -> [4, 469, 4]
            # conv xy from [4, 426, 4] to [4, 426, 1]

        xy = F.relu(self.drugsens_conv1(xy)) #[4, 32, 1] # if time stamp used is 4, then the len_out would change into 1
        fp_phAB = torch.cat([fp_phA, fp_phB], dim=-1)  # ([4, 935, 2])
        fp_phAB = F.relu(self.drugs_conv2(fp_phAB)) #[4, 32, 1]
        
        xy = xy.squeeze(2)  #[4, 32]
        fp_phAB = fp_phAB.squeeze(2) #[4, 32]
        xy_fp_phAB = torch.cat((xy, fp_phAB), 1)  #[4, 64]
        xy_fp_phAB = F.relu(self.pheno_fc1(xy_fp_phAB)) #[4, 32]
        xy_fp_phAB = self.pheno_fc2(xy_fp_phAB) #[4, 1]
        pheno = torch.sigmoid(xy_fp_phAB).squeeze(1) #[4, 1] -> [4]
        
        return y, pheno, emb_ode_conv
    
# model = ppODE(node_feats=1, pert_feats=1, hidden_feats=128, out_feats=1, pro_feats=469, drug_feature_feats=935)
