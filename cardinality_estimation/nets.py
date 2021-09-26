import torch
from torch import nn
import torch.nn.functional as F
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleRegression(torch.nn.Module):
    def __init__(self, input_width, n_output,
            num_hidden_layers,
            hidden_layer_size):
        super(SimpleRegression, self).__init__()

        self.layers = nn.ModuleList()
        layer1 = nn.Sequential(
            nn.Linear(input_width, hidden_layer_size, bias=True),
            nn.ReLU()
        ).to(device)

        self.layers.append(layer1)

        for i in range(0,num_hidden_layers-1,1):
            layer = nn.Sequential(
                nn.Linear(hidden_layer_size, hidden_layer_size, bias=True),
                nn.ReLU()
            ).to(device)
            self.layers.append(layer)

        final_layer = nn.Sequential(
            nn.Linear(hidden_layer_size, n_output, bias=True),
            nn.Sigmoid()
        ).to(device)
        self.layers.append(final_layer)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

# minor modifications on the MSCN model in Kipf et al.
class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, flow_feats,
            hid_units, num_hidden_layers=2):
        super(SetConv, self).__init__()
        self.sample_feats = sample_feats
        self.predicate_feats = predicate_feats
        self.join_feats = join_feats
        self.flow_feats = flow_feats
        num_layer1_blocks = 0

        if self.sample_feats != 0:
            self.sample_mlp1 = nn.Linear(sample_feats, hid_units).to(device)
            self.sample_mlp2 = nn.Linear(hid_units, hid_units).to(device)
            num_layer1_blocks += 1

        if self.predicate_feats != 0:
            self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units).to(device)
            self.predicate_mlp2 = nn.Linear(hid_units, hid_units).to(device)
            num_layer1_blocks += 1

        if self.join_feats != 0:
            self.join_mlp1 = nn.Linear(join_feats, hid_units).to(device)
            self.join_mlp2 = nn.Linear(hid_units, hid_units).to(device)
            num_layer1_blocks += 1

        if flow_feats != 0:
            self.flow_mlp1 = nn.Linear(flow_feats, hid_units).to(device)
            self.flow_mlp2 = nn.Linear(hid_units, hid_units).to(device)
            num_layer1_blocks += 1

        self.out_mlp1 = nn.Linear(hid_units * num_layer1_blocks,
                hid_units).to(device)
        self.out_mlp2 = nn.Linear(hid_units, 1).to(device)

    def forward(self, samples, predicates, joins, flows,
                    sample_mask, predicate_mask, join_mask):
        '''
        #TODO: describe shapes
        '''
        tocat = []
        if self.sample_feats != 0:
            samples = samples.to(device, non_blocking=True)
            sample_mask = sample_mask.to(device, non_blocking=True)
            hid_sample = F.relu(self.sample_mlp1(samples))
            hid_sample = F.relu(self.sample_mlp2(hid_sample))
            hid_sample = hid_sample * sample_mask
            hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
            sample_norm = sample_mask.sum(1, keepdim=False)
            hid_sample = hid_sample / sample_norm
            hid_sample = hid_sample.squeeze()
            tocat.append(hid_sample)

        if self.predicate_feats != 0:
            predicates = predicates.to(device, non_blocking=True)
            predicate_mask = predicate_mask.to(device, non_blocking=True)
            hid_predicate = F.relu(self.predicate_mlp1(predicates))
            hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
            hid_predicate = hid_predicate * predicate_mask
            hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
            predicate_norm = predicate_mask.sum(1, keepdim=False)
            hid_predicate = hid_predicate / predicate_norm
            hid_predicate = hid_predicate.squeeze()
            tocat.append(hid_predicate)

        if self.join_feats != 0:
            joins = joins.to(device, non_blocking=True)
            join_mask = join_mask.to(device, non_blocking=True)
            hid_join = F.relu(self.join_mlp1(joins))
            hid_join = F.relu(self.join_mlp2(hid_join))
            hid_join = hid_join * join_mask
            hid_join = torch.sum(hid_join, dim=1, keepdim=False)
            join_norm = join_mask.sum(1, keepdim=False)
            hid_join = hid_join / join_norm
            hid_join = hid_join.squeeze()
            tocat.append(hid_join)

        if self.flow_feats:
            flows = flows.to(device, non_blocking=True)
            hid_flow = F.relu(self.flow_mlp1(flows))
            hid_flow = F.relu(self.flow_mlp2(hid_flow))
            tocat.append(hid_flow)

        hid = torch.cat(tocat, 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out
