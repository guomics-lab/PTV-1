import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdyn.core import NeuralODE

class FullyConnectedLayer(nn.Module):
    """
    Custom fully connected layer with activation and dropout
    """
    def __init__(self, in_feats:int, out_feats:int, activation, dropout:float, bias:bool=True):
        super().__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self.norm = nn.LayerNorm(out_feats)  # Add layer normalization
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
        h = self.norm(h)  # Apply normalization before activation
        if self.activation:
            h = self.activation(h)
        return h

class ppODE(nn.Module):
    """
    Neural ODE model for temporal proteome and phenotype prediction
    """
    def __init__(self, node_feats, pert_feats, hidden_feats, out_feats, pro_feats, drug_feature_feats, dropout=0.1):
        super(ppODE, self).__init__()
        self.mid_feats = 32
        self.mid_feats_drugsens = 32
        self.mid_feats_drugs = 32
        self.mid_feats_drugsens_drugs = 32
        self.pro_feats = pro_feats
        self.dropout = dropout

        # Input layer
        self.linear_input = FullyConnectedLayer(in_feats=2, out_feats=self.mid_feats, activation=nn.Softplus(), dropout=dropout)

        # Convolution layers
        self.conv1 = nn.Conv1d(self.mid_feats, hidden_feats, kernel_size=1)  # Kernel size is 1 for 1D data
        self.conv1_norm = nn.GroupNorm(1, hidden_feats)  # Group normalization for conv layers
        self.conv2 = nn.Conv1d(hidden_feats, self.mid_feats, kernel_size=1)  # Kernel size is 1 for 1D data

        # Drug convolution
        self.convdrug1 = nn.Conv1d(2, hidden_feats, kernel_size=2)  # Kernel size is 2 for drug fingerprint pairs

        # NeuralODE function
        func = nn.Sequential(
            FullyConnectedLayer(in_feats=hidden_feats, out_feats=hidden_feats, activation=nn.Softplus(), dropout=dropout),
            FullyConnectedLayer(in_feats=hidden_feats, out_feats=hidden_feats, activation=None, dropout=dropout)
        )

        # NeuralODE
        self.neuralDE = NeuralODE(func, solver='rk4')

        # Final fully connected layer for protein expression
        self.layer_final = nn.Linear(self.mid_feats, out_feats)

        # Time stamp settings
        self.time_tick_num = 4

        # Phenotype prediction layers
        self.drugsens_conv1 = nn.Conv1d(self.pro_feats, self.mid_feats_drugsens, kernel_size=4)
        self.drugs_conv2 = nn.Conv1d(drug_feature_feats, self.mid_feats_drugs, kernel_size=2)
        self.pheno_fc1 = nn.Linear(self.mid_feats_drugsens + self.mid_feats_drugs, self.mid_feats_drugsens_drugs)
        self.pheno_fc2 = nn.Linear(self.mid_feats_drugsens_drugs, 1)

    def set_time_stamp_predict_drug(self, time_stamp_predict_drug):
        """
        Set the time stamp for drug prediction and initialize corresponding layers

        Args:
            time_stamp_predict_drug: Time stamp setting ('6', '24', '48', or 'all')
        """
        if time_stamp_predict_drug in ['6', '24', '48']:
            self.drugsens_conv1 = nn.Conv1d(self.pro_feats, self.mid_feats_drugsens, kernel_size=2)
        else:
            self.drugsens_conv1 = nn.Conv1d(self.pro_feats, self.mid_feats_drugsens, kernel_size=self.time_tick_num)

        # Move layer to same device as model
        self.drugsens_conv1 = self.drugsens_conv1.to(next(self.parameters()).device)

    def forward(self, x, pert, fp_phA, fp_phB, time_stamp_predict_drug):
        """
        Forward pass through the model

        Args:
            x: Protein expression data [batch_size, protein_count, 1]
            pert: Perturbation data [batch_size, protein_count, 1]
            fp_phA: Drug A fingerprint data [batch_size, fingerprint_length, 1]
            fp_phB: Drug B fingerprint data [batch_size, fingerprint_length, 1]
            time_stamp_predict_drug: Proteomic time points used for phenotype prediction

        Returns:
            y: Predicted protein expression at future time points
            pheno: Predicted drug-efficacy or combination-synergy probability
            emb_ode_conv: Embedding features from the ODE model
        """
        # Concatenate input features and embed
        emb_xpert = torch.cat([x, pert], dim=-1)  # Shape [batch_size, pro_feats, 2]
        emb_combined = self.linear_input(emb_xpert)  # Shape [batch_size, pro_feats, mid_feats]

        # Transpose for convolutional layer
        emb_combined = torch.transpose(emb_combined, 1, 2)  # [batch_size, mid_feats, pro_feats]

        # Apply convolution with normalization
        emb_cnn = self.conv1(emb_combined)  # [batch_size, hidden_feats, pro_feats]
        emb_cnn = self.conv1_norm(emb_cnn)  # Apply normalization
        emb_cnn = F.relu(emb_cnn)

        # Transpose back for ODE
        emb_cnn = torch.transpose(emb_cnn, 1, 2)  # [batch_size, pro_feats, hidden_feats]

        # Apply Neural ODE
        emb_ode = self.neuralDE(emb_cnn, torch.linspace(0, self.time_tick_num-1, self.time_tick_num))
        # Extract trajectory (skip initial condition)
        emb_ode = emb_ode[1][1:]  # [time_tick_num-1, batch_size, pro_feats, hidden_feats]

        # Reshape and transpose for second convolution
        emb_ode = torch.transpose(emb_ode, -2, -1)  # [time_tick_num-1, batch_size, hidden_feats, pro_feats]
        emb_ode = torch.transpose(emb_ode, 0, 1)  # [batch_size, time_tick_num-1, hidden_feats, pro_feats]
        emb_ode_reshape = emb_ode.reshape(-1, emb_ode.size(-2), emb_ode.size(-1))  # [batch_size*time_tick_num-1, hidden_feats, pro_feats]

        # Apply second convolution
        emb_ode_conv = self.conv2(emb_ode_reshape)  # [batch_size*time_tick_num-1, mid_feats, pro_feats]
        emb_ode_conv = torch.transpose(emb_ode_conv, -2, -1)  # [batch_size*time_tick_num-1, pro_feats, mid_feats]

        # Reshape back to include time dimension
        emb_ode_reshape_2 = emb_ode_conv.view(emb_ode.size(0), emb_ode.size(1), emb_ode_conv.size(1), emb_ode_conv.size(2))
        # [batch_size, time_tick_num-1, pro_feats, mid_feats]

        # Final layer to get protein expression predictions
        y = self.layer_final(emb_ode_reshape_2)  # [batch_size, time_tick_num-1, pro_feats, 1]

        # Time point selection for phenotype prediction
        if time_stamp_predict_drug == '6':
            xy = torch.cat([x, y[:, 0, :, :].squeeze(-1)], dim=-1)  # Use 6h time point
        elif time_stamp_predict_drug == '24':
            xy = torch.cat([x, y[:, 1, :, :].squeeze(-1)], dim=-1)  # Use 24h time point
        elif time_stamp_predict_drug == '48':
            xy = torch.cat([x, y[:, 2, :, :].squeeze(-1)], dim=-1)  # Use 48h time point
        else:
            # Use all time points
            xy = torch.cat([x, torch.transpose(y, -3, -1).squeeze(1)], dim=-1) #torch.Size([batch_size, pro_feats, 4])

        # Apply drug sensitivity convolution
        xy = F.relu(self.drugsens_conv1(xy))  # [batch_size, mid_feats_drugsens, output_size]

        # Process drug fingerprints
        fp_phAB = torch.cat([fp_phA, fp_phB], dim=-1)  # [batch_size, fingerprint_len, 2]
        fp_phAB = F.relu(self.drugs_conv2(fp_phAB))  # [batch_size, mid_feats_drugs, output_size]

        # Squeeze dimensions
        xy = xy.squeeze(2)  # [batch_size, mid_feats_drugsens]
        fp_phAB = fp_phAB.squeeze(2)  # [batch_size, mid_feats_drugs]

        # Concatenate protein and drug features
        xy_fp_phAB = torch.cat((xy, fp_phAB), 1)  # [batch_size, mid_feats_drugsens+mid_feats_drugs]

        # Final layers for phenotype prediction
        xy_fp_phAB = F.relu(self.pheno_fc1(xy_fp_phAB))  # [batch_size, mid_feats_drugsens_drugs]
        xy_fp_phAB = self.pheno_fc2(xy_fp_phAB)  # [batch_size, 1]
        pheno = torch.sigmoid(xy_fp_phAB).squeeze(1)  # [batch_size]

        return y, pheno, emb_ode_conv
