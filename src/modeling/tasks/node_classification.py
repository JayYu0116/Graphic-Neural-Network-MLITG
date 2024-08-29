import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.gcn import GCN


class NodeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super(NodeClassifier, self).__init__()

        self.gcn = GCN(input_dim=input_dim, 
                       hidden_dim=hidden_dim, 
                       output_dim= hidden_dim, 
                       dropout=dropout)# TODO: initialize the GCN model
        self.dropout_feat=nn.Dropout(dropout)
        self.node_classifier = nn.Linear(hidden_dim,n_classes)# TODO: initialize the linear classifier
        self.activation0=torch.nn.ReLU()
        self.final=torch.nn.LogSoftmax(dim=-1)
    def forward(
        self, x: torch.Tensor, adj: torch.sparse_coo, classify: bool = True
    ) -> torch.Tensor:
        # TODO: implement the forward pass of the node classification task
        if classify:
            x=self.gcn(x,adj)
            x=self.node_classifier(x)
            #x=torch.nn.functional.softmax(x,dim=-1)
            x=self.final(x)
        else:
            x=self.gcn(x,adj)
        return x
        
