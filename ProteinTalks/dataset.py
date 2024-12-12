import torch
import numpy as np
import pandas as pd
import re
from utils import *


class L1000Dataset(torch.utils.data.Dataset):
    def __init__(self, cellline, data_dir):
        loo_label_name = True
        nodes = pd.read_csv(data_dir + "/" + cellline + "node_Index.csv", header=None)
        expr = pd.read_csv(data_dir + "/" + cellline + "expr.csv", header=None)
        drug_fp_phychemA = pd.read_csv(data_dir + "/" + cellline + "drug_fp_phychem_A.csv", header=None)
        drug_fp_phychemB = pd.read_csv(data_dir + "/" + cellline + "drug_fp_phychem_B.csv", header=None)
        drug_fp_phychemA = drug_fp_phychemA.applymap(str_to_list)
        drug_fp_phychemB = drug_fp_phychemB.applymap(str_to_list)

        # deltamid = pd.read_csv(data_dir + "/" + cellline + "_deltamid.csv", header=None)
        # deltamax = pd.read_csv(data_dir + "/" + cellline + "_deltamax.csv", header=None)
        
        if loo_label_name:
            loo_label = pd.read_csv(data_dir + "/" + cellline + "loo_label.csv", header=None)
            timepoint_x, timepoint_y1, timepoint_y2, timepoint_y3 = "0", "6", "24", "48"
        pert = pd.read_csv(data_dir + "/" + cellline + "pert.csv", header=None)
        pheno = pd.read_csv(data_dir + "/" + cellline + "pheno.csv", header=None)
        
        self.x_data = []
        self.y_data = []
        self.pert_data = []
        self.nodes = nodes
        self.pheno = []
        self.xdruga = []
        self.xdrugb = []
        # self.deltamid = []
        # self.deltamax = []
        
        all_experiment_type_redu = np.sort(list(set(loo_label[0]))) #make sure the order is the same
        all_experiment_type = [i for i in all_experiment_type_redu if "#" in i] #cellline + drug, eg 'BT20_#69 #37'
        times_to_check = [6, 24, 48]
        for experiment_type in all_experiment_type:
            if (len(loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == 6)])!=0)&(len(loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == 24)])!=0)&(len(loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == 48)])!=0):
                if (expr.loc[loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == int(48))].index[0]]).any():
                    timepoint = timepoint_x
                    if loo_label_name:
                        pattern = re.escape(experiment_type.split('_')[0]) #only cellline
                        x_values = expr.loc[loo_label[(loo_label[0].str.contains(pattern)) & (loo_label[1] == int(timepoint))].index[0]].values[0:]
                    else:
                        x_values = expr.loc[loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == int(timepoint))].index[0]].values[0:]
                    x = torch.tensor(np.log(x_values+1)).float().unsqueeze(1)
                    x_min = torch.min(x)
                    x_max = torch.max(x)
                    x_norm = (x - x_min) / (x_max - x_min)
                    self.x_data.append(x_norm)
                    
                    # Set node features based on timepoint "6" as y_data
                    y_norm_6_24_48 = []
                    for timepoint_y in [timepoint_y1, timepoint_y2, timepoint_y3]:
                        timepoint = timepoint_y
                        y_values = expr.loc[loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == int(timepoint))].index[0]].values[0:]
                        y = torch.tensor(np.log(y_values+1)).float().unsqueeze(1)
                        y_min = torch.min(y)
                        y_max = torch.max(y)
                        y_norm = (y - y_min) / (y_max - y_min)
                        y_norm_6_24_48.append(y_norm)
                    self.y_data.append(torch.stack(y_norm_6_24_48, dim=0))
                    
                    # Add drug features as xdrug
                    xdruga = drug_fp_phychemA.loc[loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == int(timepoint))].index[0]].values[0] #药物信息细化
                    xdrugb = drug_fp_phychemB.loc[loo_label[(loo_label[0] == experiment_type) & (loo_label[1] == int(timepoint))].index[0]].values[0] #药物信息细化
                    try:
                        xdruga = torch.tensor(xdruga).float().unsqueeze(1)
                        xdrugb = torch.tensor(xdrugb).float().unsqueeze(1)
                    except:
                        print(experiment_type)
                        print(xdruga)
                    xdruga_min = torch.min(xdruga)
                    xdruga_max = torch.max(xdruga)
                    xdruga_norm = (xdruga - xdruga_min) / (xdruga_max - xdruga_min)
                    xdrugb_min = torch.min(xdrugb)
                    xdrugb_max = torch.max(xdrugb)
                    xdrugb_norm = (xdrugb - xdrugb_min) / (xdrugb_max - xdrugb_min)
                    self.xdruga.append(xdruga_norm) #shape [935, 1]
                    self.xdrugb.append(xdrugb_norm) #shape [935, 1]
                    
                    # Add perturbations as pert_data
                    pert_values = pert.loc[loo_label[loo_label[0] == experiment_type].index[0]].values[0:] 
                    
                    pert_tensor = torch.tensor(pert_values).float().unsqueeze(1)
                    self.pert_data.append(pert_tensor)
                    
                    # Add pheonotype data
                    pheno_values = float(pheno.loc[loo_label[loo_label[0] == experiment_type].index[0]].values)
            
                    # Modified for the avoid of negtive inf of pheno_values
                    if np.isnan(pheno_values):
                        pheno_tensor = torch.tensor(0.0).float()
                    else:
                        pheno_tensor = torch.tensor(pheno_values).float()
                    self.pheno.append(pheno_tensor)
                    
                    # deltamid_row = deltamid.loc[loo_label[(loo_label[0].str.contains(pattern)) & (loo_label[1] == int(timepoint))].index[0]].values[0]
                    # deltamax_row = deltamax.loc[loo_label[(loo_label[0].str.contains(pattern)) & (loo_label[1] == int(timepoint))].index[0]].values[0]
                    # self.deltamid.append(torch.tensor(deltamid_row).float())
                    # self.deltamax.append(torch.tensor(deltamax_row).float())
            
    def __getitem__(self, idx):
        return self.x_data[idx], self.pert_data[idx], self.y_data[idx], self.pheno[idx], self.xdruga[idx], self.xdrugb[idx]
    
    def __len__(self):
        return len(self.x_data)
    
    def get_protein_name(self):
        return self.nodes
    
    # def get_deltamid(self, idx):
    #     return torch.tensor([self.deltamid[idx], self.deltamax[idx]])
    
    # def get_deltamax(self, idx):
    #     return self.deltamax[idx]
    