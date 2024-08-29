import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.gcn import GCN


class LinkPrediction(nn.Module):
    def __init__(self,  hidden_dim: int):
        """Link prediction module.
        We want to predict the edge label (0 or 1) for each edge.
        We assume that the model gets the node features as input (i.e., GCN is already applied to the node features).
        Args:
            hidden_dim (int): [The hidden dimension of the GCN layer (i.e., feature dimension of the nodes)]
        """
        
        super(LinkPrediction, self).__init__()

        self.edge_classifier = nn.Sequential(torch.nn.Linear(2*hidden_dim,4*hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(4*hidden_dim,2))
        self.act=nn.ReLU()
        self.node1_project=nn.Identity(hidden_dim,hidden_dim) #project feat1 to feat1prime
        self.node2_project=nn.Identity(hidden_dim,hidden_dim)#project feat2 to feat1prime
        #self.edge_classifier=torch.nn.Linear(2*hidden_dim,2)
        self.dropout=nn.Dropout(0)
        #self.activation=nn.LogSoftmax(dim=-1)
    def forward(
        self, node_features_after_gcn: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        first_node_feat=self.node1_project(node_features_after_gcn[edges[0],:])#(edge,feat)
        second_node_feat=self.node2_project(node_features_after_gcn[edges[1],:])
        edge_feat=torch.cat((first_node_feat,second_node_feat),dim=1)#(edge,feat)
        edge_feat=self.act(edge_feat)#ReLU
        x=self.edge_classifier(edge_feat)

        return x

