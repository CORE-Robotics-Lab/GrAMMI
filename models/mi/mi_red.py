import torch
import math
import torch.nn as nn


class RedMI(nn.Module):
    def __init__(self, input_dim, output_dim, bayesian_embedding_dim = 4, h1=16, h2=32, non_linear=nn.ReLU()):
        super(RedMI, self).__init__()
        self.input_dim = input_dim + bayesian_embedding_dim
        self.h1 = h1
        self.h2 = h2
        self.bayesian_embedding_dim = bayesian_embedding_dim
        self.output_dim = output_dim
        self.non_linear = non_linear
        # self.init_bayesian_embedding()

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, h1),
            self.non_linear,
            nn.Linear(h1, h2),
            self.non_linear,
            nn.Linear(h2, output_dim),
            )
        # initialize our layers
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, embedding_tensor):
        """ x is of shape (batch_size, input_dim)
        embedding_tensor is of shape (batch_size, bayesian_embedding_dim) which holds parameters """
        # predict the next location of the agent
        ins = torch.cat((x, embedding_tensor), dim=1)
        return self.network(ins)

    