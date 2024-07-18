"""
This script demonstrates the implementation of a simple Graph Neural Network (GNN)
using PyTorch Geometric. It includes functions for generating synthetic graph data,
training a model, evaluating its performance, and conducting nested cross-validation
with hyperparameter tuning using Optuna.

The script covers the setup of the GNN model, the training and evaluation process,
and uses Optuna for optimizing the GNN hyperparameters through a nested cross-validation
approach to assess model robustness and parameter effectiveness.
"""

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.model_selection import KFold
import numpy as np
import networkx as nx
import logging
import optuna

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_graph_data(num_graphs, num_nodes, num_features):
    """
    Generates synthetic graph data with random features and labels using the Erdos-Renyi model.
    
    Args:
        num_graphs (int): Number of graphs to generate.
        num_nodes (int): Number of nodes per graph.
        num_features (int): Number of features per node.
    
    Returns:
        list of Data: List containing the generated graph data.
    """
    graphs = []
    for _ in range(num_graphs):
        G = nx.erdos_renyi_graph(n=num_nodes, p=0.5)
        x = torch.randn((num_nodes, num_features))
        y = torch.tensor([np.random.rand()])
        edge_index = torch.tensor(list(G.edges)).t().contiguous() if G.edges else torch.empty((2, 0), dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs

class SimpleGNN(torch.nn.Module):
    """
    A simple Graph Neural Network module using two GCNConv layers followed by a global mean pooling layer and a linear output layer.
    """
    def __init__(self, num_features, hidden_channels):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.out(x)

def train_model(model, loader, optimizer, criterion):
    """
    Trains a given model using the specified data loader, optimizer, and loss criterion.
    
    Args:
        model (torch.nn.Module): The model to train.
        loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer used for model training.
        criterion (Loss): Loss function used for training.
    
    Returns:
        float: Average training loss over all batches.
    """
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        out = out.view(-1)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader):
    """
    Evaluates a given model using the specified data loader.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader containing the test data.
    
    Returns:
        float: Average error over all test data.
    """
    model.eval()
    total_error = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            out = out.view(-1)
            error = F.l1_loss(out, data.y)
            total_error += error.item()
    return total_error / len(loader)

def objective(trial, train_loader, val_loader, num_features):
    """
    Objective function for hyperparameter tuning using Optuna. Trains and validates the model using suggested parameters.
    
    Args:
        trial (Trial): Optuna trial object that suggests hyperparameters.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_features (int): Number of features in the input data.
    
    Returns:
        float: Validation error for the trial.
    """
    hidden_channels = trial.suggest_int('hidden_channels', 16, 64)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    epochs = trial.suggest_int('epochs', 10, 50)
    
    model = SimpleGNN(num_features, hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out.view(-1), data.y)
            loss.backward()
            optimizer.step()

    model.eval()
    total_error = 0
    with torch.no_grad():
        for data in val_loader:
            out = model(data)
            loss = criterion(out.view(-1), data.y)
            total_error += loss.item()

    average_error = total_error / len(val_loader)
    return average_error

def nested_cross_validation(graphs, num_features, num_splits_outer=5, num_splits_inner=3):
    """
    Performs nested cross-validation on the given graphs to evaluate model performance and hyperparameter tuning.
    
    Args:
        graphs (list of Data): Graph data for cross-validation.
        num_features (int): Number of features in each graph.
        num_splits_outer (int): Number of splits for outer cross-validation.
        num_splits_inner (int): Number of splits for inner cross-validation.
    """
    outer_kfold = KFold(n_splits=num_splits_outer, shuffle=True, random_state=42)
    all_test_errors = []

    for train_idx, test_idx in outer_kfold.split(graphs):
        train_data = [graphs[i] for i in train_idx]
        test_data = [graphs[i] for i in test_idx]

        best_params = None
        best_validation_error = float('inf')

        inner_kfold = KFold(n_splits=num_splits_inner, shuffle=True, random_state=42)
        for inner_train_idx, inner_val_idx in inner_kfold.split(train_data):
            inner_train_data = DataLoader([train_data[i] for i in inner_train_idx], batch_size=10, shuffle=True)
            inner_val_data = DataLoader([train_data[i] for i in inner_val_idx], batch_size=10, shuffle=False)

            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, inner_train_data, inner_val_data, num_features), n_trials=20)

            if study.best_trial.value < best_validation_error:
                best_validation_error = study.best_trial.value
                best_params = study.best_trial.params

        final_model = SimpleGNN(num_features, best_params['hidden_channels'])
        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
        final_criterion = torch.nn.MSELoss()
        final_train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

        for _ in range(best_params['epochs']):
            train_model(final_model, final_train_loader, final_optimizer, final_criterion)

        test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
        test_error = evaluate_model(final_model, test_loader)
        all_test_errors.append(test_error)
        logging.info(f'Test Error: {test_error}')

    logging.info(f'Average Test Error: {np.mean(all_test_errors)}, Std Dev: {np.std(all_test_errors)}')

def main():
    num_graphs = 100
    graphs = generate_graph_data(num_graphs, num_nodes=10, num_features=5)
    nested_cross_validation(graphs, num_features=5)

if __name__ == "__main__":
    main()
