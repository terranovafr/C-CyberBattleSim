# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    gae_utils.py
    This file contains the utility functions for the GAE model training and validation.
"""

import torch
import torch.nn.functional as F
import os
import sys
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import DataLoader
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
torch.set_default_dtype(torch.float32)
from cyberbattle.utils.math_utils import diversity_loss # noqa: E402
from cyberbattle.utils.encoding_utils import convert_privilege_to_indices, convert_status_to_indices # noqa: E402


# Training function for the model simply forward pass and loss computation
def compute_backward_batch_train_loss(model, batch, optimizer, adj_weight=1,
          feature_weight=1, edge_feature_weight=1, diversity_weight=1, binary_cat_weight=1, multi_cat_weight=1, cont_weight=1, **config):
    model.train()
    optimizer.zero_grad()

    total_loss, adj_loss, feature_loss, edge_feature_loss, div_loss, binary_cat_loss, multi_cat_loss, cont_loss = compute_loss(model, batch, adj_weight=adj_weight, feature_weight=feature_weight, edge_feature_weight=edge_feature_weight, diversity_weight=diversity_weight, binary_cat_weight=binary_cat_weight, multi_cat_weight=multi_cat_weight, cont_weight=cont_weight, **config)
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), adj_loss.item(), feature_loss.item(), edge_feature_loss.item(), div_loss.item(), binary_cat_loss.item(), multi_cat_loss.item(), cont_loss.item()


# Overall loss function used during training and validation
def compute_loss(model, data, binary_indices, multi_class_info, continuous_indices, adj_weight=1,
          feature_weight=1, edge_feature_weight=1, diversity_weight=1, binary_cat_weight=1, multi_cat_weight=1, cont_weight=1, **config):
    filtered_embeddings = []
    for embedding in data.vulnerabilities_embeddings:
        filtered_embeddings.append(embedding)
    if len(filtered_embeddings) == 0: # If no edges are present (graph with just discovered nodes) add a fictious set of edges with zero embeddings
        filtered_embeddings = [torch.zeros(config['edge_feature_vector_size'], device=data.x.device)]

    data.vulnerabilities_embeddings = torch.stack(filtered_embeddings).float()
    data.x = data.x.float()

    # Determine the vector of reconstructed node features, adjacency matrix, and edge features by the model
    reconstructed_x, reconstructed_adj, reconstructed_edge_attr = model(data.x, data.edge_index, data.vulnerabilities_embeddings)

    # Adjacency matrix computation and loss calculation
    adj = torch.zeros((data.num_nodes, data.num_nodes), device=data.x.device)
    adj[data.edge_index[0], data.edge_index[1]] = 1  # Binary adjacency matrix
    adj_loss = F.binary_cross_entropy_with_logits(reconstructed_adj, adj)

    # Handle categorical and continuous features differently for feature vector reconstruction
    multi_cat_loss = torch.tensor(0.0, device=data.x.device)

    # Use binary cross-entropy loss for binary features
    if binary_indices:
        binary_cat_loss = F.binary_cross_entropy_with_logits(
            reconstructed_x[:, binary_indices], data.x[:, binary_indices])
    else:
        binary_cat_loss = torch.tensor(0, device=data.x.device)

    # Use multi-class cross-entropy loss for multi-class features
    offset = len(binary_indices)
    offset_ground_truth = len(binary_indices)
    for idx, num_classes in multi_class_info.items():
        logits = reconstructed_x[:, offset:offset + num_classes]
        if idx == "privilege level":
            ground_truth = convert_privilege_to_indices(data.x[:, offset_ground_truth])
        elif idx == "status":
            ground_truth = convert_status_to_indices(data.x[:, offset_ground_truth])
        else:
            ground_truth = data.x[:, offset_ground_truth].long()
        multi_cat_loss += F.cross_entropy(logits, ground_truth)
        offset += num_classes
        offset_ground_truth += 1

    # Use mean squared error loss for continuous features
    cont_loss = F.mse_loss(reconstructed_x[:, continuous_indices], data.x[:, continuous_indices])

    # Weight elements of the feature vector loss
    feature_loss = binary_cat_weight * binary_cat_loss + multi_cat_weight * multi_cat_loss + cont_weight * cont_loss

    # Diversity loss among node embeddings
    div_loss = diversity_loss(reconstructed_x)

    # Use mean squared error loss for edge features
    if edge_feature_weight > 0:
        edge_feature_loss = F.mse_loss(reconstructed_edge_attr, data.vulnerabilities_embeddings)
        if torch.isnan(edge_feature_loss).any():
            edge_feature_loss = torch.tensor(0, device=data.x.device)
    else:
        edge_feature_loss = torch.tensor(0, device=data.x.device)

    # Combine all losses with proper weights
    total_loss = (
            adj_weight * adj_loss + feature_weight * feature_loss + edge_feature_weight * edge_feature_loss + diversity_weight * div_loss)

    return total_loss, adj_loss, feature_loss, edge_feature_loss, div_loss, binary_cat_loss, multi_cat_loss, cont_loss


# Validation function: periodically use and switch the validation set of graphs
def validate(model, val_env, writer, config, starting_epoch):
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    total_adj_loss = 0
    total_feature_loss = 0
    total_edge_feature_loss = 0
    total_diversity_loss = 0
    total_binary_cat_loss = 0
    total_multi_cat_loss = 0
    total_cont_loss = 0
    count = 0
    done = True
    batch = []
    for iteration in range(config['val_iterations']):
        if done:
            val_env.reset()
        G = val_env.current_env.evolving_visible_graph
        data = from_networkx(G)

        # Sample a valid action and advance the graph to ensure to see another configuration
        source_node, target_node, vulnerability_ID, outcome_desired = val_env.sample_valid_action()
        done = val_env.step_graph(source_node, target_node, vulnerability_ID, outcome_desired)
        if 'vulnerabilities_embeddings' not in data: # If no edges are present (graph with just discovered nodes) add a fictious set of edges with zero embeddings
            data.vulnerabilities_embeddings = torch.zeros((data.num_edges, config['edge_feature_vector_size']), device=data.x.device)
        batch.append(data)
        if len(batch) % config['batch_size'] == 0:
            batch_loader = DataLoader(batch, batch_size=config['batch_size'], shuffle=True)
            # turn batch loader into iterable so that I can do for loop in it
            for batch_data in batch_loader:  # type: ignore
                with torch.no_grad():  # No gradients needed for validation step
                    total_loss, adj_loss, feature_loss, edge_feature_loss, div_loss, binary_cat_loss, multi_cat_loss, cont_loss = compute_loss(model, batch_data, **config)
                    total_val_loss += total_loss.item()
                    total_adj_loss += adj_loss.item()
                    total_feature_loss += feature_loss.item()
                    if not torch.isnan(edge_feature_loss).any():
                        total_edge_feature_loss += edge_feature_loss.item()
                    total_diversity_loss += div_loss.item()
                    total_cont_loss += cont_loss.item()
                    total_binary_cat_loss += binary_cat_loss
                    total_multi_cat_loss += multi_cat_loss
                    count += 1

    # Across all validation iterations, calculate the average losses
    average_val_loss = total_val_loss / count if count else 0
    average_adj_loss = total_adj_loss / count if count else 0
    average_feature_loss = total_feature_loss / count if count else 0
    average_edge_feature_loss = total_edge_feature_loss / count if count else 0
    average_diversity_loss = total_diversity_loss / count if count else 0
    average_cont_loss = total_cont_loss / count if count else 0
    average_binary_cat_loss = total_binary_cat_loss / count if count else 0
    average_multi_cat_loss = total_multi_cat_loss / count if count else 0

    # Log the average validation losses to TensorBoard
    writer.add_scalar('val/total_loss', average_val_loss, starting_epoch+config['val_iterations'])
    if config['weights']['adj_weight'] > 0:
        writer.add_scalar('val/adj_loss', average_adj_loss, starting_epoch+config['val_iterations'])
    if config['weights']['node_feature_vector_weight'] > 0:
        writer.add_scalar('val/node_feature_vector_loss', average_feature_loss, starting_epoch+config['val_iterations'])
    if config['weights']['edge_feature_vector_weight'] > 0:
        writer.add_scalar('val/edge_feature_vector_loss', average_edge_feature_loss, starting_epoch+config['val_iterations'])
    if config['weights']['diversity_weight'] > 0:
        writer.add_scalar('val/diversity_loss', average_diversity_loss, starting_epoch+config['val_iterations'])
    if config['weights']['node_feature_vector_binary_cat_weight'] > 0:
        writer.add_scalar('val/node_feature_vector/binary_cat_loss', average_binary_cat_loss, starting_epoch+config['val_iterations'])
    if config['weights']['node_feature_vector_multi_cat_weight'] > 0:
        writer.add_scalar('val/node_feature_vector/multi_cat_loss', average_multi_cat_loss, starting_epoch+config['val_iterations'])
    if config['weights']['node_feature_vector_cont_weight'] > 0:
        writer.add_scalar('val/node_feature_vector/cont_loss', average_cont_loss, starting_epoch+config['val_iterations'])

    model.train()  # Set the model back to training mode

    return average_val_loss, average_adj_loss, average_feature_loss, average_edge_feature_loss, average_diversity_loss, average_binary_cat_loss, average_multi_cat_loss, average_cont_loss
