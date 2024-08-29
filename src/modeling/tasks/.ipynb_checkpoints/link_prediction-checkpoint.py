import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.gcn import GCN


class LinkPrediction(nn.Module):
    def __init__(self, hidden_dim: int):
        """Link prediction module.
        We want to predict the edge label (0 or 1) for each edge.
        We assume that the model gets the node features as input (i.e., GCN is already applied to the node features).
        Args:
            hidden_dim (int): [The hidden dimension of the GCN layer (i.e., feature dimension of the nodes)]
        """
        
        super(LinkPrediction, self).__init__()
        self.edge_classifier = nn.Sequential(torch.nn.Linear(2*hidden_dim,4*hidden_dim),
                                             nn.Linear(4*hidden_dim,2)) 
        self.activation=nn.Softmax(dim=-1)
    def forward(
        self, node_features_after_gcn: torch.Tensor, edges: torch.Tensor,
    ) -> torch.Tensor:
        first_node_feat=node_features_after_gcn[edges[0],:]#(edge,feat)
        second_node_feat=node_features_after_gcn[edges[1],:]
        edge_feat=torch.cat((first_node_feat,second_node_feat),dim=1)#(edge,feat)
        x=self.edge_classifier(edge_feat)
        x=self.activation(x)
        # node_features_after_gcn: [num_nodes, hidden_dim]
        # edges: [2, num_edges]
        # the function should return classifier logits for each edge
        # Note that the output should not be probabilities, rather one logit for each class (so the output should be batch_size x 2).
        # TODO: Implement the forward pass of the link prediction module
        return x

