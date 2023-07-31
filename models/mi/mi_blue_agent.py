import torch
import math

import dgl
import dgl.nn as dglnn
import torch.nn as nn

from dgl.nn import AvgPooling
from models.encoders import EncoderRNN
from torch.distributions import Normal
from models.utils import log_prob
from itertools import combinations

def fully_connected(num_nodes):
    # create fully connected graph
    test_list = range(num_nodes)
    edges = list(combinations(test_list, 2))
    start_nodes = [i[0] for i in edges]
    end_nodes = [i[1] for i in edges]
    return torch.tensor(start_nodes), torch.tensor(end_nodes)


class BlueMIMixtureGNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures,  h1, h2, gnn_hidden, non_linear=nn.ReLU()):
        super(BlueMIMixtureGNN, self).__init__()

        self.input_dim = input_dim
        self.h1 = h1
        self.h2 = h2
        self.gnn_hidden = gnn_hidden
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        self.non_linear = non_linear

        network_input_dim = self.h1 + self.gnn_hidden + num_mixtures

        self.obs_encoder = EncoderRNN(input_dim, h1)
        self.agent_lstm = EncoderRNN(3, h1)

        self.act = nn.LeakyReLU()
        self.conv1 = dglnn.SAGEConv(
            in_feats=h1, out_feats=h2, aggregator_type='pool', activation=self.act)
        self.conv2 = dglnn.SAGEConv(
            in_feats=h2, out_feats=gnn_hidden, aggregator_type='pool')
        
        self.avgpool = AvgPooling()
        self.initialized_graphs = {n: self.initialize_graph(n) for n in range(3, 8)}

        self.fc = nn.Linear(network_input_dim, h1)
        
        self.pi = nn.Linear(h1, 1)
        nn.init.xavier_normal_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)

        self.mu = nn.Linear(h1, output_dim)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

        self.sigma = nn.Linear(h1, output_dim)
        nn.init.xavier_normal_(self.sigma.weight)
        nn.init.zeros_(self.sigma.bias)

        self.softmax = nn.Softmax(dim=1)

    def initialize_graph(self, num_agents):
        # initialize graph just from 
        s, e = fully_connected(num_agents)
        s = s.to(self.device)
        e = e.to(self.device)
        g = dgl.graph((s, e))
        g = dgl.add_reverse_edges(dgl.to_homogeneous(g))
        return g

    def encode(self, x):
        obs, agents, num_agents = x
        z = self.obs_encoder(obs)

        batch_size = z.shape[0]
        graph_list = []
        for n in num_agents:
            n = int(n.item())
            g = self.initialized_graphs[n]
            graph_list.append(g)

        batched_graphs = dgl.batch(graph_list).to(self.device)
        hn = self.agent_lstm(agents)

        res = self.conv1(batched_graphs, hn)
        res = self.conv2(batched_graphs, res)
        res = self.avgpool(batched_graphs, res)

        z = torch.cat((z, res), dim=1)
        return z

    def forward(self, x):
        pi_categoricals = torch.eye(self.num_mixtures).to(self.device)
        pis, mus, sigmas = [], [], []
        # z = self.encoder(x)
        z = self.encode(x)
        batch_size = z.shape[0]
        

        for i in range(self.num_mixtures):
            pi_cat = pi_categoricals[i].unsqueeze(0).repeat(batch_size, 1)
            ins = torch.cat((z, pi_cat), dim=1)
            ins = self.fc(ins)
            ins = self.non_linear(ins)

            
            pi = self.pi(ins) # B x 1
            mu = self.mu(ins) # B x 2

            sigma = self.sigma(ins) # B x 2
            sigma = torch.exp(sigma) # B x 2
            sigma = nn.ELU()(sigma) + 1e-15

            pis.append(pi)
            mus.append(mu)
            sigmas.append(sigma)

        pis = torch.cat(pis, dim=-1) # B x num_mixtures x 1
        pis = self.softmax(pis) # B x num_mixtures x 1
        mus = torch.stack(mus, dim=1) # B x num_mixtures x 2
        sigmas = torch.stack(sigmas, dim=1) # B x num_mixtures x 2
        
        return pis, mus, sigmas

    def forward_encoder(self, x):
        pi_categoricals = torch.eye(self.num_mixtures).to(self.device)
        pis, mus, sigmas = [], [], []
        # z = self.encoder(x)
        z = self.encode(x)
        batch_size = z.shape[0]
        

        pi_cats = []
        for i in range(self.num_mixtures):
            pi_cat = pi_categoricals[i].unsqueeze(0).repeat(batch_size, 1)
            ins = torch.cat((z, pi_cat), dim=1)
            ins = self.fc(ins)
            ins = self.non_linear(ins)
            
            pi = self.pi(ins) # B x 1
            mu = self.mu(ins) # B x 2

            sigma = self.sigma(ins) # B x 2
            sigma = torch.exp(sigma) # B x 2
            sigma = nn.ELU()(sigma) + 1e-15

            pis.append(pi)
            mus.append(mu)
            sigmas.append(sigma)

            pi_cats.append(pi_cat)

        p = torch.cat(pis, dim=0)
        m = torch.cat(mus, dim=0)
        s = torch.cat(sigmas, dim=0)
        all_outputs = torch.cat((p, m, s), dim=1)


        pis = torch.cat(pis, dim=-1) # B x num_mixtures x 1
        pis = self.softmax(pis) # B x num_mixtures x 1
        mus = torch.stack(mus, dim=1) # B x num_mixtures x 2
        sigmas = torch.stack(sigmas, dim=1) # B x num_mixtures x 2
        
        pi_cats = torch.cat(pi_cats, dim=0) # B x num_mixtures x 3

        z_posterior = z.repeat((self.num_mixtures, 1))

        return (pis, mus, sigmas), (z_posterior, all_outputs), pi_cats


    def forward_embeds(self, x, embeds):
        """ Given a batch of embeddings, predict the location of the agent"""
        # z = self.encoder(x)
        z = self.encode(x)
        batch_size = z.shape[0]

        ins = torch.cat((z, embeds), dim=1)
        ins = self.fc(ins)
        ins = self.non_linear(ins)
        
        # pi = self.pi(ins) # B x 1
        mu = self.mu(ins) # B x 2
        sigma = self.sigma(ins) # B x 2
        sigma = torch.exp(sigma) # B x 2
        sigma = nn.ELU()(sigma) + 1e-15

        return mu, sigma

    def compute_loss(self, x, red_locs):
        nn_output = self.forward(x)
        red_locs = red_locs.to(self.device)
        return mdn_negative_log_likelihood_loss(nn_output, red_locs)

    @property
    def device(self):
        return next(self.parameters()).device


def mdn_negative_log_likelihood(pi, mu, sigma, target):
    """ Use torch.logsumexp for more stable training 
    
    This is equivalent to the mdn_loss but computed in a numerically stable way

    """
    target = target.unsqueeze(1).expand_as(sigma)
    # target = target.unsqueeze(2).expand_as(sigma)
    neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
        ((target - mu) / sigma)**2 / 2
    
    inner = torch.log(pi) + torch.sum(neg_logprob, 2) # Sum the log probabilities of (x, y) for each 2D Gaussian
    return -torch.logsumexp(inner, dim=1)

def mdn_negative_log_likelihood_loss(nn_output, target):
    """
    Compute the negative log likelihood loss for a MoG model.
    """
    pi, mu, sigma = nn_output
    return mdn_negative_log_likelihood(pi, mu, sigma, target).mean()