import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.layers import GCNLayer


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super(GCN, self).__init__()
        # TODO: add 2 layers of GCN
        self.GCN0=GCNLayer(in_features=input_dim, 
                           out_features=hidden_dim)
        self.GCN1=GCNLayer(in_features=hidden_dim, 
                           out_features=output_dim)
        self.activation0=nn.ReLU()
        self.activation1=nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.sparse_coo) -> torch.Tensor:
        # given the input node features, and the adjacency matrix, run GCN
        # The order of operations should roughly be:
        # 1. Apply the first GCN layer
        x=self.GCN0(x,adj)
        
        
        
        # 2. Apply Relu
        x=self.activation0(x)
        # 3. Apply Dropout
        x=self.dropout(x)
        # 4. Apply the second GCN layer
        x=self.GCN1(x,adj)
        # TODO: your code here
        #x=self.activation1(x)
        return x
