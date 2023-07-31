# rather than the heterogeneous lstm - let's use the lstm in the front and then pass these into a heterogeneous
# graph
import torch
import torch.nn as nn
from models.encoders import EncoderRNN
import torch

import torch
import dgl
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn.conv.transformer_conv import TransformerConv

def construct_het_graph(num_agents, num_hideouts):
    """ Script to construct heterogeneous graph from dataset 
    
    We have a graph with node types: agent, hideout, hideout summary node, agent summary node, timestep node
    
    
    agent -> agent summary node
    hideout -> hideout summary node
    
    hideout summary, agent summary -> state summary

    """

    # from agent to agent summary node
    agent_indices = torch.arange(0, num_agents)
    agent_summary_index = torch.tensor([0] * num_agents) # torch.zeros didn't work - error with dgl

    hideout_indices = torch.arange(0, num_hideouts)
    hideout_summary_index = torch.tensor([0] * num_hideouts)
    # dgl datadict is from (from_node_index_tensor, to_node_index_tensor) where different node types are indexed differently

    data_dict = {
        ('agent', 'to', 'agent_summ'): (agent_indices, agent_summary_index),
        ('hideout', 'to', 'hideout_summ'): (hideout_indices, hideout_summary_index),
        ('hideout_summ', 'to', 'state_summ'): (torch.tensor([0]), torch.tensor([0])),
        ('agent_summ', 'to', 'state_summ') : (torch.tensor([0]), torch.tensor([0])),
    }

    return dgl.heterograph(data_dict)

def convert_batched_dgl_graph_to_pyg(bg):
    """ Converts a batch of dgl graphs to single pytorch geometric heterodata graph """
    data=HeteroData()
    edge_list = [('agent', 'to', 'agent_summ'), ('hideout', 'to', 'hideout_summ'), ('hideout_summ', 'to', 'state_summ'), ('agent_summ', 'to', 'state_summ')]

    for edge in edge_list:
        data[edge[0], edge[1], edge[2]].edge_index = torch.stack(bg.edges(etype=edge))

    return data

# Create graph encoder consisting of an lstm into a graph NN
class LSTMHeteroPost(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_hidden, num_layers=1):
        super(LSTMHeteroPost, self).__init__()
        self.hidden_dim = hidden_dim
        self.initialized_graphs = dict()
        self.lstm = EncoderRNN(input_dim, hidden_dim, num_layers)

        self.act = nn.Tanh()

        self.metadata = (['agent', 'hideout', 'timestep', 'agent_summ', 'hideout_summ', 'state_summ'], 
        [('agent', 'to', 'agent_summ'), 
        ('hideout', 'to', 'hideout_summ'), 
        ('hideout_summ', 'to', 'state_summ'), 
        ('agent_summ', 'to', 'state_summ'), 
        ('agent_summ', 'rev_to', 'agent'), 
        ('hideout_summ', 'rev_to', 'hideout'), 
        ('state_summ', 'rev_to', 'hideout_summ'), 
        ('state_summ', 'rev_to', 'agent_summ')])

        # in_channels_dict = {'agent': hidden_dim, 'hideout': 2, 'agent_summ': 1, 'hideout_summ': 1, 'state_summ': 1}
        self.conv1 = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                                    out_channels=hidden_dim,
                                                    bias=True) for edge_type in self.metadata[1]})
        self.conv2 = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                                    out_channels=gnn_hidden,
                                                    bias=True) for edge_type in self.metadata[1]})
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        # x is (batch_size, seq_len, num_agents, features)
        # input_tensor = torch.randn((batch_size, seq_len, num_agents, features))

        # agent_obs, hideout_obs, timestep_obs = x
        
        agent_obs, hideout_obs, timestep_obs, num_agents = x
        agent_obs = agent_obs.to(self.device).float()

        # agent_obs = torch.cat((agent_obs, location_obs), dim=2)

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()

        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        # num_agents = agent_obs.shape[2]
        features = agent_obs.shape[3]

        # permuted = agent_obs.permute(0, 2, 1, 3) # (batch_size, num_agents, seq_len, features)
        # hn is of shape (batch_size * num_agents, hidden_dim)
        # .view(batch_size * lstm_input.shape[1], seq_len, features)

        graph_list = []
        lstm_input = []
        for n, batch in zip(num_agents, list(range(batch_size))):
            n = int(n.item())

            if n not in self.initialized_graphs:
                dgl_graph = construct_het_graph(n, 1)
                self.initialized_graphs[n] = dgl_graph
            else:
                dgl_graph = self.initialized_graphs[n]

            h = agent_obs[batch, :, :n, :]
            lstm_input.append(h)
            graph_list.append(dgl_graph)

        batched_graphs = dgl.batch(graph_list)
        pyg = convert_batched_dgl_graph_to_pyg(batched_graphs).to(self.device)
        pyg = T.ToUndirected()(pyg)

        lstm_input = torch.cat(lstm_input, dim=1).contiguous()
        lstm_input = lstm_input.permute(1, 0, 2)
        hn = self.lstm(lstm_input)

        pyg['agent'].x = hn
        pyg['hideout'].x = hideout_obs # hideouts don't change over time
        pyg['agent_summ'].x = torch.zeros((batch_size, 1), device=self.device)
        pyg['hideout_summ'].x = torch.zeros((batch_size, 1), device=self.device)
        pyg['state_summ'].x = torch.zeros((batch_size, 1), device=self.device)

        # res = self.conv1(batched_graphs, hn)
        out_dict = self.conv1(pyg.x_dict, pyg.edge_index_dict)
        res = self.conv2(out_dict, pyg.edge_index_dict)
        res = {node_type: self.act(feat) for node_type, feat in res.items() }
        # res = torch.cat((res['state_summ'], hideout_obs, timestep_obs), dim=-1)
        return res['state_summ']