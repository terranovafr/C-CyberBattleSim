# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    model.py
    This file contains the model used for the GAE model trained using unsupervised learning.
    The model is composed by a GNN-based encoder and a NN-based decoder, with the first using node feature vectors, topology features, and edge features to encode the graph structure,
     and the second reconstructing several elements in order to ensure the graph structure is preserved.
"""

import torch
import os
from torch_geometric.nn import GCNConv, GATConv, EdgeConv, NNConv
from torch.nn import ReLU, ModuleList, Sequential, Linear
import sys
from torch_geometric.nn.norm import BatchNorm
import torch.nn as nn
import torch.nn.functional as F
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
torch.set_default_dtype(torch.float32)

# Encoder GNN-based architecture used to encode the graph structure
class GAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, cfg_layers, edge_feature_dim):
        super(GAEEncoder, self).__init__()
        self.layers = ModuleList()
        # Custom number of layers of several types
        for layer_cfg in cfg_layers:
            if layer_cfg['type'] == 'GCNConv':
                self.layers.append(GCNConv(in_channels, layer_cfg['out_channels']))
                in_channels = layer_cfg['out_channels']
            elif layer_cfg['type'] == 'GATConv':
                self.layers.append(GATConv(in_channels, layer_cfg['out_channels'], heads=layer_cfg.get('heads', 1),
                                           concat=layer_cfg.get('concat', True)))
                in_channels = layer_cfg['out_channels'] * layer_cfg.get('heads', 1) if layer_cfg.get('concat',
                                                                                                     True) else \
                layer_cfg['out_channels']
            elif layer_cfg['type'] == 'EdgeConv':
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(2 * in_channels, layer_cfg['out_channels']),
                    torch.nn.ReLU(),
                    torch.nn.Linear(layer_cfg['out_channels'], layer_cfg['out_channels'])
                )
                self.layers.append(EdgeConv(mlp, aggr=layer_cfg.get('aggr', 'mean')))
                in_channels = layer_cfg['out_channels']
            elif layer_cfg['type'] == 'NNConv':
                # this layer ensures also usage of edge features
                edge_network = Sequential(
                    Linear(edge_feature_dim, layer_cfg['NN_channels']),
                    ReLU(),
                    Linear(layer_cfg['NN_channels'], in_channels * layer_cfg['out_channels'])
                )

                self.layers.append(NNConv(in_channels, layer_cfg['out_channels'], edge_network))
                in_channels = layer_cfg['out_channels']
            self.layers.append(BatchNorm(in_channels))
            activation = layer_cfg.get('activation', 'ReLU')
            if activation == 'ReLU':
                self.layers.append(ReLU())
            elif activation == 'Sigmoid':
                self.layers.append(torch.nn.Sigmoid())
            elif activation == 'Tanh':
                self.layers.append(torch.nn.Tanh())
            elif activation == 'LeakyReLU':
                self.layers.append(torch.nn.LeakyReLU())
            # otherwise not adding any activation

    def forward(self, x, edge_index, edge_attr=None):
        # Forwarding based on the type of layers
        for layer in self.layers:
            if isinstance(layer, (GCNConv, GATConv)):
                x = layer(x, edge_index)
            elif isinstance(layer, NNConv):
                if edge_attr is not None:
                    x = layer(x, edge_index, edge_attr)
            elif isinstance(layer, EdgeConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

# NN-based decoder used to reconstruct characteristics based on node embeddings
class GAEDecoder(nn.Module):
    def __init__(self, out_channels, edge_feat_dim, binary_indices, multi_class_info, continuous_indices):
        super(GAEDecoder, self).__init__()

        # Setup for binary, multi-class categorical data, and continuous data
        self.binary_indices = binary_indices
        self.multi_class_info = multi_class_info
        self.continuous_indices = continuous_indices

        # Calculate the output sizes for each type of data
        self.num_binary = len(binary_indices)
        self.num_multi_class = sum(multi_class_info.values())

        # Decoder for binary categorical features
        if self.num_binary > 0:
            self.binary_feature_decoder = nn.Sequential(
                nn.Linear(out_channels, self.num_binary), # one output neuron per binary feature
                nn.Sigmoid() # sigmoid applied to ensure values are between 0 and 1
            )

        # Decoder for multi-class categorical features
        if self.num_multi_class > 0:
            self.multi_class_feature_decoder = nn.Linear(out_channels, self.num_multi_class) # one output neuron per each class summing up to the total number of classes

        # Decoder for continuous features
        self.cont_feature_decoder = nn.Sequential(
            nn.Linear(out_channels, len(continuous_indices))
        )

        # Decoder for adjacency matrix
        self.adj_decoder = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid() # Sigmoid applied to ensure values are between 0 and 1
        )

        # Decoder for edge features
        self.edge_feat_decoder = nn.Sequential(
            nn.Linear(2 * out_channels, edge_feat_dim), # takes the concatenation of the two node embeddings
        )

    def forward(self, z, edge_index):
        # Initialize output tensor for all features
        total_outputs = self.num_binary + self.num_multi_class + len(self.continuous_indices)
        reconstructed_x = torch.zeros((z.size(0), total_outputs), device=z.device)

        # Decode binary features
        if self.num_binary > 0:
            binary_features = self.binary_feature_decoder(z)
            reconstructed_x[:, :self.num_binary] = binary_features

        # Decode multi-class features
        offset = self.num_binary
        if self.num_multi_class > 0:
            multi_class_logits = self.multi_class_feature_decoder(z)
            # Softmax applied segment-wise for each categorical feature
            start = 0
            for idx, num_classes in self.multi_class_info.items():
                end = start + num_classes
                reconstructed_x[:, offset + start:offset + end] = F.softmax(multi_class_logits[:, start:end], dim=1)
                start += num_classes

        # Decode continuous features
        continuous_features = self.cont_feature_decoder(z)
        reconstructed_x[:, -len(self.continuous_indices):] = continuous_features

        # Reconstruct adjacency matrix
        adj = torch.mm(z, z.t())
        edge_embeddings = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)

        # Reconstruct edge features
        edge_features = self.edge_feat_decoder(edge_embeddings)
        return reconstructed_x, adj, edge_features

# Overall GAE combining the encoder and decoder
class GAE(torch.nn.Module):
    def __init__(self, in_channels, cfg_layers, edge_feat_dim, binary_indices, multi_class_info, continuous_indices):
        super(GAE, self).__init__()
        self.encoder = GAEEncoder(in_channels, cfg_layers, edge_feat_dim)
        last_layer_output_channels = cfg_layers[-1]['out_channels']
        self.decoder = GAEDecoder(last_layer_output_channels, edge_feat_dim, binary_indices, multi_class_info, continuous_indices)

    def forward(self, x, edge_index, edge_attr):
        z = self.encoder(x, edge_index, edge_attr)  # Node embeddings
        reconstructed_x, reconstructed_adj, reconstructed_edge_features = self.decoder(z, edge_index)  # Reconstructed elements
        return reconstructed_x, reconstructed_adj, reconstructed_edge_features
