import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.core.layers import GCNLayer
LAYER=3

class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super(GCN, self).__init__()
        # TODO: add 2 layers of GCN
        if LAYER==2:
            self.layers=torch.nn.ModuleList([GCNLayer(in_features=input_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=output_dim)]
                                        )
        elif LAYER==3:
            self.layers=torch.nn.ModuleList([GCNLayer(in_features=input_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=output_dim)]
                                        )
        elif LAYER==4:
            self.layers=torch.nn.ModuleList([GCNLayer(in_features=input_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=output_dim)]
                                        )
        elif LAYER==5:
            self.layers=torch.nn.ModuleList([GCNLayer(in_features=input_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=output_dim)]
                                        )
        elif LAYER>=6:
            self.layers=torch.nn.ModuleList([GCNLayer(in_features=input_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        GCNLayer(in_features=hidden_dim, 
                                                    out_features=output_dim)]
                                        )
    def forward(self, x: torch.Tensor, adj: torch.sparse_coo) -> torch.Tensor:
        # given the input node features, and the adjacency matrix, run GCN
        # The order of operations should roughly be:
        # 1. Apply the first GCN layer
        for layer in self.layers:
            if isinstance(layer,GCNLayer):
                x=layer(x,adj)
            else:
                x=layer(x)

        return x
