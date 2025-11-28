import pandas as pd
import numpy as np
# import pickle
import pickle5 as pickle
import json
from rdkit import Chem
from torch_geometric.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_batch
import torch
from typing import Sequence, Tuple, List, Union
from ligand_graph_features import *
# from fingerprint.graph import *
##### tsv modules #####


def save_interaction_tsv(p,tsvname,datalist):
    with open(os.path.join(p,tsvname), 'wt',newline='') as f_output:
        tsv_output=csv.writer(f_output, delimiter='\t')
        for c in datalist:
            tsv_output.writerow(c)
def read_interaction_tsv(path):
  lines = []
  with open(path, 'r') as f:
    for line in f:
      lines.append(line.split('\n')[0].split('\t'))
  return lines

##### JSON modules #####
def save_json(data,filename):
  with open(filename, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=4)

def load_json(filename):
  with open(filename, 'r') as fp:
    data = json.load(fp)
  return data

##### pickle modules #####
def save_dict_pickle(data,filename):
  with open(filename,'wb') as handle:
    pickle.dump(data,handle, pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
  with open(path, 'rb') as f:
    dict = pickle.load(f)
  return  dict


##### DTI #####

#------------------
#  read data
#------------------

def load_training_data(exp_path):
    train = pd.read_csv(exp_path + 'train.csv')
    dev = pd.read_csv(exp_path + 'dev.csv')
    test = pd.read_csv(exp_path + 'test.csv')
    return train, dev, test

from torch_geometric.data import Batch        # 새로 import
def get_repr_DTI(batch_data, tokenizer, chem_dict, protein_dict, prot_descriptor_choice, chem_option):
    batch_data = batch_data.reset_index(drop=True)
    valid_idx, chem_graphs = [], []

    for i, ikey in enumerate(batch_data['ikey']):
        smiles = chem_dict.get(ikey)
        if not isinstance(smiles, str):
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        chem_graphs.append(mol_to_graph_data_obj_simple(mol))
        valid_idx.append(i)

    # keep only valid rows
    batch_data = batch_data.iloc[valid_idx].reset_index(drop=True)

    # re-tokenize proteins for exact same length
    uniprot_list = batch_data['uniprot'].tolist()
    protein_tokenized = torch.tensor(
        [tokenizer.encode(protein_dict[uni]) for uni in uniprot_list],
        dtype=torch.long
    )

    # build DataLoader
    loader = DataLoader(chem_graphs, batch_size=len(chem_graphs), shuffle=False)
    chem_batch = next(iter(loader))

    return batch_data, chem_batch, protein_tokenized



def get_repr_DTI_original(batch_data,tokenizer,chem_dict,protein_dict,prot_descriptor_choice,chem_option):
    #  . . . .  chemicals  . . . .
    batch_data = batch_data.reset_index(drop = True)
    chem_smiles = chem_dict[batch_data['ikey'].values.tolist()].values.tolist()
    chem_graph_list = []
    del_idx = []
    for smiles in chem_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            del_idx.append(i)
            continue
        graph = mol_to_graph_data_obj_simple(mol)
        chem_graph_list.append(graph)
    batch_data = batch_data.drop(batch_data.index[del_idx], axis = 0)

    chem_graphs_loader = DataLoader(chem_graph_list, batch_size=batch_data.shape[0],
                                    shuffle=False)
    for batch in chem_graphs_loader:
        chem_graphs = batch

    uniprot_list = batch_data['uniprot'].values.tolist()
    protein_tokenized = torch.tensor([tokenizer.encode(protein_dict[uni]) for uni in uniprot_list  ])
    return batch_data, chem_graphs, protein_tokenized



