from __future__ import print_function
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GraphDegreeConv(nn.Module):

    def __init__(self, node_size, edge_size, output_size, degree_list,
            ntype, etype, batch_normalize=True):
        super(GraphDegreeConv, self).__init__()
        self.ntype = ntype
        self.etype = etype
        self.node_size = node_size
        self.edge_size = edge_size
        self.output_size = output_size

        self.batch_normalize = batch_normalize
        if self.batch_normalize:
            self.normalize = nn.BatchNorm1d(output_size, affine=False)

        self.bias = nn.Parameter(torch.zeros(1, output_size))
        self.linear = nn.Linear(node_size, output_size, bias=False)
        self.degree_list = degree_list
        self.degree_layer_list = nn.ModuleList()
        for degree in degree_list:
            self.degree_layer_list.append(nn.Linear(node_size + edge_size, output_size, bias=False))

    def forward(self, graph, node_repr, edge_repr, neighbor_by_degree):
        logging.debug("Convolutional layer: {0}".format(self.linear))

        degree_activation_list = []
        for d_idx, degree_layer in enumerate(self.degree_layer_list):
            degree = self.degree_list[d_idx]
            node_neighbor_list = neighbor_by_degree[degree]['node']
            edge_neighbor_list = neighbor_by_degree[degree]['edge']
            if degree == 0 and node_neighbor_list:
                zero = Variable(torch.zeros(len(node_neighbor_list), self.output_size))
                if torch.cuda.is_available():
                    zero = zero.cuda()
                degree_activation_list.append(zero)
            else:
                if node_neighbor_list:
                    # (#nodes, #degree, node_size)
                    node_neighbor_repr = node_repr[node_neighbor_list, ...]
                    # (#nodes, #degree, edge_size)
                    edge_neighbor_repr = edge_repr[edge_neighbor_list, ...]
                    # (#nodes, #degree, node_size + edge_size)
                    stacked = torch.cat([node_neighbor_repr, edge_neighbor_repr], dim=2)
                    summed = torch.sum(stacked, dim=1, keepdim=False)
                    degree_activation = degree_layer(summed)
                    degree_activation_list.append(degree_activation)

        neighbor_repr = torch.cat(degree_activation_list, dim=0)
        self_repr = self.linear(node_repr)
        # size = (#nodes, #output_size)

        activations = self_repr + neighbor_repr + self.bias.expand_as(self_repr)
        if self.batch_normalize:
            activations = self.normalize(activations)
        return F.relu(activations)

class NeuralFingerprint(nn.Module):

    def __init__(self, node_size, edge_size, conv_layer_sizes, output_size, type_map,
            degree_list, batch_normalize=True):
        """
        Args:
            node_size (int): dimension of node representations
            edge_size (int): dimension of edge representations
            conv_layer_sizes (list of int): the lengths of the output vectors
                of convolutional layers
            output_size (int): length of the finger print vector
            type_map (dict string:string): type of the batch nodes, vertex nodes,
                and edge nodes
            degree_list (list of int): a list of degrees for different
                convolutional parameters
            batch_normalize (bool): enable batch normalization (default True)
        """
        super(NeuralFingerprint, self).__init__()
        self.num_layers = len(conv_layer_sizes)
        self.output_size = output_size
        self.batch_type = type_map['batch']
        self.ntype = type_map['node']
        self.etype = type_map['edge']
        self.degree_list = degree_list

        self.conv_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        layers_sizes = [node_size] + conv_layer_sizes
        for input_size in layers_sizes:
            self.out_layers.append(nn.Linear(input_size, output_size))
        for prev_size, next_size in zip(layers_sizes[:-1], layers_sizes[1:]):
            self.conv_layers.append(
                GraphDegreeConv(prev_size, edge_size, next_size, degree_list,
                                self.ntype, self.etype, batch_normalize=batch_normalize))

    def forward(self, graph):
        """
        Args:
            graph (Graph): A graph object that represents a mini-batch
        Returns:
            fingerprint: A tensor variable with shape (batch_size, output_size)
        """
        batch_size = len(graph.get_node_list(self.batch_type))

        logging.debug("Initiating variables for {0} nodes...".format(len(graph.get_node_list(self.ntype))))
        # node_repr.size = (#nodes, #features)
        node_repr = Variable(torch.FloatTensor([node.data for node in graph.get_node_list(self.ntype)]))
        edge_repr = Variable(torch.FloatTensor([node.data for node in graph.get_node_list(self.etype)]))
        # fingerprint.size = (batch_size, output_size)
        fingerprint = Variable(torch.zeros(batch_size, self.output_size))
        if torch.cuda.is_available():
            node_repr = node_repr.cuda()
            edge_repr = edge_repr.cuda()
            fingerprint = fingerprint.cuda()

        logging.debug("Collecting neighbor indices...")
        neighbor_by_degree = []
        for degree in self.degree_list:
            neighbor_by_degree.append({
                'node': graph.get_neighbor_idx_by_degree(self.ntype, self.ntype, degree),
                'edge': graph.get_neighbor_idx_by_degree(self.ntype, self.etype, degree)
            })
        batch_idx = graph.get_neighbor_idx(self.batch_type, self.ntype)

        def fingerprint_update(linear, node_repr):
            logging.debug("Updating fingerprint...")
            logging.debug("Vector: {0}:{1}, Layer: {2}".format(node_repr.size(), type(node_repr.data), linear))
            atom_activations = F.softmax(linear(node_repr),dim=1)
            logging.debug("atom size: {0}".format(atom_activations.size()))
            update = torch.cat([torch.sum(atom_activations[atom_idx, ...], dim=0, keepdim=True) for atom_idx in batch_idx], dim=0)
            return update

        for layer_idx in range(self.num_layers):
            # (#nodes, #output_size)
            fingerprint += fingerprint_update(self.out_layers[layer_idx], node_repr)
            node_repr = self.conv_layers[layer_idx](graph, node_repr, edge_repr, neighbor_by_degree)
        fingerprint += fingerprint_update(self.out_layers[-1], node_repr)

        # Unsqueeze a pseudo-length=1 so that down stream functions
        # can stay unchanged
        fingerprint = fingerprint.unsqueeze(1)
        logging.debug("Fingerprint shape: {}".format(fingerprint.size()))

        return fingerprint
