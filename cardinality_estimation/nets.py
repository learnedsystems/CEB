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
        x = x.to(device, non_blocking=True)
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

# minor modifications on the MSCN model in Kipf et al.
class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, flow_feats,
            hid_units, num_hidden_layers=2, n_out=1,
            dropouts=[0.0, 0.0, 0.0], use_sigmoid=True):
        super(SetConv, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.sample_feats = sample_feats
        self.predicate_feats = predicate_feats
        self.join_feats = join_feats
        self.flow_feats = flow_feats
        num_layer1_blocks = 0

        self.inp_drop = dropouts[0]
        self.hl_drop = dropouts[1]
        self.combined_drop = dropouts[2]
        self.inp_drop_layer = nn.Dropout(p=self.inp_drop)
        self.hl_drop_layer = nn.Dropout(p=self.hl_drop)
        self.combined_drop_layer = nn.Dropout(p=self.combined_drop)

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

        # if flow_feats != 0:
            # self.flow_mlp1 = nn.Linear(flow_feats, hid_units).to(device)
            # self.flow_mlp2 = nn.Linear(hid_units, hid_units).to(device)
            # num_layer1_blocks += 1

        combined_hid_size = hid_units + flow_feats

        comb_size = (hid_units * num_layer1_blocks) + flow_feats

        self.out_mlp1 = nn.Linear(comb_size,
                combined_hid_size).to(device)

        # unless flow_feats is 0
        combined_hid_size += flow_feats
        self.out_mlp2 = nn.Linear(combined_hid_size, n_out).to(device)

    def forward(self, xbatch):
        '''
        #TODO: describe shapes
        '''
        samples = xbatch["table"]
        predicates = xbatch["pred"]
        joins = xbatch["join"]
        flows = xbatch["flow"]

        sample_mask = xbatch["tmask"]
        predicate_mask = xbatch["pmask"]
        join_mask = xbatch["jmask"]

        tocat = []
        if self.sample_feats != 0:
            samples = samples.to(device, non_blocking=True)
            sample_mask = sample_mask.to(device, non_blocking=True)
            samples = self.inp_drop_layer(samples)
            hid_sample = F.relu(self.sample_mlp1(samples))
            hid_sample = self.hl_drop_layer(hid_sample)

            hid_sample = F.relu(self.sample_mlp2(hid_sample))
            hid_sample = hid_sample * sample_mask
            hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)

            if torch.sum(sample_mask) == 0:
                hid_sample = torch.zeros(hid_sample.shape).squeeze()
            else:
                sample_norm = sample_mask.sum(1, keepdim=False)
                hid_sample = hid_sample / sample_norm
                hid_sample = hid_sample.squeeze()

            tocat.append(hid_sample)

        if self.predicate_feats != 0:
            predicates = predicates.to(device, non_blocking=True)
            predicate_mask = predicate_mask.to(device, non_blocking=True)
            predicates = self.inp_drop_layer(predicates)

            hid_predicate = F.relu(self.predicate_mlp1(predicates))
            hid_predicate = self.hl_drop_layer(hid_predicate)

            hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
            hid_predicate = hid_predicate * predicate_mask
            hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
            predicate_norm = predicate_mask.sum(1, keepdim=False)
            hid_predicate = hid_predicate / predicate_norm
            hid_predicate = hid_predicate.squeeze()
            tocat.append(hid_predicate)

        if self.join_feats != 0:
            joins = joins.to(device, non_blocking=True)
            joins = self.inp_drop_layer(joins)
            join_mask = join_mask.to(device, non_blocking=True)

            hid_join = F.relu(self.join_mlp1(joins))
            hid_join = self.hl_drop_layer(hid_join)

            hid_join = F.relu(self.join_mlp2(hid_join))
            hid_join = hid_join * join_mask
            hid_join = torch.sum(hid_join, dim=1, keepdim=False)

            if torch.sum(join_mask) == 0:
                hid_join = torch.zeros(hid_join.shape).squeeze()
            else:
                join_norm = join_mask.sum(1, keepdim=False)
                hid_join = hid_join / join_norm
                hid_join = hid_join.squeeze()

            tocat.append(hid_join)


        if self.flow_feats:
            flows = flows.to(device, non_blocking=True)
            flows = self.inp_drop_layer(flows)
            # hid_flow = F.relu(self.flow_mlp1(flows))
            # hid_flow = self.hl_drop_layer(hid_flow)
            # hid_flow = F.relu(self.flow_mlp2(hid_flow))
            tocat.append(flows)

        hid = torch.cat(tocat, 1)
        hid = self.combined_drop_layer(hid)
        hid = F.relu(self.out_mlp1(hid))
        if self.flow_feats:
            hid = torch.cat([hid, flows], 1)

        if self.use_sigmoid:
            out = torch.sigmoid(self.out_mlp2(hid))
        else:
            out = self.out_mlp2(hid)
        return out

class SetConvFlow(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, flow_feats,
            hid_units, num_hidden_layers=2, n_out=1,
            dropouts=[0.0, 0.0, 0.0], use_sigmoid=True):
        super(SetConvFlow, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.sample_feats = sample_feats
        self.predicate_feats = predicate_feats
        self.join_feats = join_feats
        self.flow_feats = flow_feats
        num_layer1_blocks = 0

        self.inp_drop = dropouts[0]
        self.hl_drop = dropouts[1]
        self.combined_drop = dropouts[2]
        self.inp_drop_layer = nn.Dropout(p=self.inp_drop)
        self.hl_drop_layer = nn.Dropout(p=self.hl_drop)
        self.combined_drop_layer = nn.Dropout(p=self.combined_drop)

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

        comb_size = (hid_units * num_layer1_blocks)

        self.out_mlp1 = nn.Linear(comb_size,
                hid_units).to(device)

        self.out_mlp2 = nn.Linear(hid_units, n_out).to(device)

    def forward(self, xbatch):
        '''
        #TODO: describe shapes
        '''
        samples = xbatch["table"]
        predicates = xbatch["pred"]
        joins = xbatch["join"]
        flows = xbatch["flow"]

        sample_mask = xbatch["tmask"]
        predicate_mask = xbatch["pmask"]
        join_mask = xbatch["jmask"]

        tocat = []
        if self.sample_feats != 0:
            samples = samples.to(device, non_blocking=True)
            sample_mask = sample_mask.to(device, non_blocking=True)
            samples = self.inp_drop_layer(samples)
            hid_sample = F.relu(self.sample_mlp1(samples))
            hid_sample = self.hl_drop_layer(hid_sample)

            hid_sample = F.relu(self.sample_mlp2(hid_sample))
            hid_sample = hid_sample * sample_mask
            hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)

            if torch.sum(sample_mask) == 0:
                hid_sample = torch.zeros(hid_sample.shape).squeeze()
            else:
                sample_norm = sample_mask.sum(1, keepdim=False)
                hid_sample = hid_sample / sample_norm
                hid_sample = hid_sample.squeeze()

            tocat.append(hid_sample)

        if self.predicate_feats != 0:
            predicates = predicates.to(device, non_blocking=True)
            predicate_mask = predicate_mask.to(device, non_blocking=True)
            predicates = self.inp_drop_layer(predicates)

            hid_predicate = F.relu(self.predicate_mlp1(predicates))
            hid_predicate = self.hl_drop_layer(hid_predicate)

            hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
            hid_predicate = hid_predicate * predicate_mask
            hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
            predicate_norm = predicate_mask.sum(1, keepdim=False)
            hid_predicate = hid_predicate / predicate_norm
            hid_predicate = hid_predicate.squeeze()
            tocat.append(hid_predicate)

        if self.join_feats != 0:
            joins = joins.to(device, non_blocking=True)
            joins = self.inp_drop_layer(joins)
            join_mask = join_mask.to(device, non_blocking=True)

            hid_join = F.relu(self.join_mlp1(joins))
            hid_join = self.hl_drop_layer(hid_join)

            hid_join = F.relu(self.join_mlp2(hid_join))
            hid_join = hid_join * join_mask
            hid_join = torch.sum(hid_join, dim=1, keepdim=False)

            if torch.sum(join_mask) == 0:
                hid_join = torch.zeros(hid_join.shape).squeeze()
            else:
                join_norm = join_mask.sum(1, keepdim=False)
                hid_join = hid_join / join_norm
                hid_join = hid_join.squeeze()

            tocat.append(hid_join)


        if self.flow_feats:
            flows = flows.to(device, non_blocking=True)
            flows = self.inp_drop_layer(flows)
            hid_flow = F.relu(self.flow_mlp1(flows))
            hid_flow = self.hl_drop_layer(hid_flow)
            hid_flow = F.relu(self.flow_mlp2(hid_flow))
            tocat.append(hid_flow)

        hid = torch.cat(tocat, 1)
        hid = self.combined_drop_layer(hid)
        hid = F.relu(self.out_mlp1(hid))

        # if self.flow_feats:
            # hid = torch.cat([hid, flows], 1)

        if self.use_sigmoid:
            out = torch.sigmoid(self.out_mlp2(hid))
        else:
            out = self.out_mlp2(hid)
        return out
