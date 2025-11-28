import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from copy import deepcopy

# Local imports (Must be present in the directory)
from model_Yang import * 
from fp_features import num_atom_features, num_bond_features
from fp_models import NeuralFingerprint
from resnet import ResnetEncoderModel
from data_tool_box import *

class DeepREALModel(nn.Module):
    """
    Main DeepREAL Model for 3-class classification (Agonist, Antagonist, Non-binder)
    Previously named DTI_3_class_HJ
    """
    def __init__(self, all_config=None, DTI_binary_pretrained=None):
        super(DeepREALModel, self).__init__()
        self.all_config = all_config
        self.DTI_binary_pretrained = deepcopy(DTI_binary_pretrained)
        
        print('Initializing Attentive Pooler for Multi-class...')
        self.attentive_interaction_pooler = AttentivePooling(300) # Assuming hidden size matches input
        
        # Interaction Pooler dimensions based on config 'choice'
        # Defaulting logic for benchmark clarity
        input_dim = 300 + 256 # Default
        
        self.interaction_pooler = EmbeddingTransform(input_dim, 128, 64, 0.1)
        
        # Determine choice dimension
        choice_dim = 64
        if all_config.get('choice') == 'binary_embed_new_interaction_pipe':
            choice_dim = 64
        elif all_config.get('choice') == 'binary_embed_new_interaction_pipe+binary_embed':
            choice_dim = 620
        elif all_config.get('choice') == 'all':
            self.multi_learner = ResnetEncoderModel(1)
            self.multi_classifer = EmbeddingTransform(112, 64, 3)
            self.DTI_binary_pretrained_NO_FORGETTING = deepcopy(DTI_binary_pretrained)
            return # Exit early for 'all' case

        self.multi_classifer = EmbeddingTransform(choice_dim, 64, 3)

    def forward(self, batch_protein_tokenized, batch_chem_graphs, epoch):
        # Freeze encoder after a certain epoch
        if epoch > self.all_config['frozen_epoch']:
            with torch.no_grad():
                batch_chem_repr, batch_prot_repr = self.DTI_binary_pretrained.embed2(
                    batch_protein_tokenized, batch_chem_graphs)
        else:
            batch_chem_repr, batch_prot_repr = self.DTI_binary_pretrained.embed2(
                batch_protein_tokenized, batch_chem_graphs)

        # Attentive Pooling
        ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(
            batch_chem_repr, batch_prot_repr
        ) 
        
        # Interaction Vector
        inter_vect3 = self.interaction_pooler(torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))
        
        # Classifier
        logits = self.multi_classifer(inter_vect3)
        return logits


class DTI_model_pretrained(nn.Module):
    """
    Pretrained binary DTI model wrapper.
    """
    def __init__(self, all_config=None,
                 contextpred_config={'num_layer': 5, 'emb_dim': 300, 'JK': 'last', 'drop_ratio': 0.5, 'gnn_type': 'gin'},
                 model=None):
        super(DTI_model_pretrained, self).__init__()
        self.use_cuda = all_config['use_cuda']
        self.contextpred_config = contextpred_config
        self.all_config = all_config

        # 1. Ligand Embedding
        if all_config['chem_option'] == 'contextpred':
            self.ligandEmbedding = GNN(
                num_layer=contextpred_config['num_layer'],
                emb_dim=contextpred_config['emb_dim'],
                JK=contextpred_config['JK'],
                drop_ratio=contextpred_config['drop_ratio'],
                gnn_type=contextpred_config['gnn_type']
            )
        else:
            self.ligandEmbedding = ChemicalGraphConv(use_cuda=self.use_cuda)

        # 2. Protein Embedding
        self.proteinEmbedding = model
        prot_embed_dim = 256 # ResNet output dim

        # Frozen logic (Partial freezing of DISAE layers)
        if all_config.get('prot_frozen') == 'partial':
            ct = 0
            frozen_list = all_config.get('DISAE', {}).get('frozen_list', [])
            for m in self.proteinEmbedding.modules():
                ct += 1
                if ct in frozen_list:
                    for param in m.parameters():
                        param.requires_grad = False
                else:
                    for param in m.parameters():
                        param.requires_grad = True

        self.resnet = ResnetEncoderModel(1)

        # 3. Interaction
        self.attentive_interaction_pooler = AttentivePooling(contextpred_config['emb_dim'])
        self.interaction_pooler = EmbeddingTransform(contextpred_config['emb_dim'] + prot_embed_dim, 128, 64, 0.1)
        self.binary_predictor = EmbeddingTransform(64, 64, 2, 0.2)

        if self.use_cuda and torch.cuda.is_available():
            self.attentive_interaction_pooler = self.attentive_interaction_pooler.to('cuda')
            self.interaction_pooler = self.interaction_pooler.to('cuda')
            self.binary_predictor = self.binary_predictor.to('cuda')
            self.ligandEmbedding = self.ligandEmbedding.to('cuda')
            self.proteinEmbedding = self.proteinEmbedding.to('cuda')

    def embed2(self, batch_protein_tokenized, batch_chem_graphs, **kwargs):
        # Protein Embedding
        batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]
        bs = batch_protein_repr.size(0)
        batch_protein_repr_resnet = self.resnet(batch_protein_repr.unsqueeze(1)).view(bs, 1, -1)

        # Ligand Embedding
        node_representation = self.ligandEmbedding(batch_chem_graphs.x, batch_chem_graphs.edge_index, batch_chem_graphs.edge_attr)
        batch_chem_graphs_repr_masked, _ = to_dense_batch(node_representation, batch_chem_graphs.batch)
        batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1).unsqueeze(1)

        return batch_chem_graphs_repr_pooled, batch_protein_repr_resnet


class EmbeddingTransform(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, dropout_p=0.1):
        super(EmbeddingTransform, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size)
        )

    def forward(self, embedding):
        embedding = self.dropout(embedding)
        hidden = self.transform(embedding)
        return hidden


class AttentivePooling(nn.Module):
    """ Attentive pooling network according to https://arxiv.org/pdf/1602.03609.pdf """
    def __init__(self, chem_hidden_size=300, prot_hidden_size=256):
        super(AttentivePooling, self).__init__()
        self.chem_hidden_size = chem_hidden_size
        self.prot_hidden_size = prot_hidden_size
        self.param = nn.Parameter(torch.zeros(chem_hidden_size, prot_hidden_size))

    def forward(self, first, second):
        if first.size(0) != second.size(0):
            raise RuntimeError(f"Batch size mismatch: chem={first.size(0)}, prot={second.size(0)}")
        
        # first: (batch, len, hidden)
        param = self.param.expand(first.size(0), self.chem_hidden_size, self.prot_hidden_size)
        
        wm1 = torch.tanh(torch.bmm(second, param.transpose(1, 2)))
        wm2 = torch.tanh(torch.bmm(first, param))

        score_m1 = F.softmax(wm1, dim=2)
        score_m2 = F.softmax(wm2, dim=2)

        rep_first = first * score_m1
        rep_second = second * score_m2

        return ((rep_first, score_m1), (rep_second, score_m2))

class ChemicalGraphConv(nn.Module):
    def __init__(self, conv_layer_sizes=[20, 20, 20, 20], output_size=300,
                 degrees=[0, 1, 2, 3, 4, 5], num_atom_features=num_atom_features(),
                 num_bond_features=num_bond_features(), use_cuda=None):
        super(ChemicalGraphConv, self).__init__()
        type_map = dict(batch='molecule', node='atom', edge='bond')
        self.model = NeuralFingerprint(
            num_atom_features, num_bond_features, conv_layer_sizes,
            output_size, type_map, degrees, use_cuda=use_cuda
        )

        for param in self.model.parameters():
            param.data.uniform_(-0.08, 0.08)

    def forward(self, batch_input, **kwargs):
        batch_embedding = self.model(batch_input)
        return batch_embedding