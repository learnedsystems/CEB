import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.functional as F
import time
import math
from .set_transformer import SetTransformer
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEBUG_TIMES=False

class SimpleRegression(torch.nn.Module):
    def __init__(self, input_width, n_output,
            num_hidden_layers,
            hidden_layer_size,
            ):
        super(SimpleRegression, self).__init__()
        hidden_layer_size = int(hidden_layer_size)

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
    def __init__(self, sample_feats, predicate_feats, join_feats,
            flow_feats,
            hid_units,
            other_hid_units,
            num_hidden_layers=2, n_out=1,
            dropouts=[0.0, 0.0, 0.0], use_sigmoid=True):
        super(SetConv, self).__init__()

        ## debug time code
        self.total_fwd_time = 0.0

        hid_units = int(hid_units)
        if other_hid_units is not None:
            other_hid_units = int(other_hid_units)

        self.use_sigmoid = use_sigmoid

        self.sample_feats = sample_feats
        self.predicate_feats = predicate_feats
        self.join_feats = join_feats
        self.flow_feats = flow_feats
        self.num_hidden_layers = num_hidden_layers
        num_layer1_blocks = 0

        self.inp_drop = dropouts[0]
        self.hl_drop = dropouts[1]
        self.combined_drop = dropouts[2]
        self.inp_drop_layer = nn.Dropout(p=self.inp_drop)
        self.hl_drop_layer = nn.Dropout(p=self.hl_drop)
        self.combined_drop_layer = nn.Dropout(p=self.combined_drop)

        self.sample_mlps = nn.ModuleList()
        self.predicate_mlps = nn.ModuleList()
        self.join_mlps = nn.ModuleList()

        if hid_units < 4:
            # gonna treat this as a multiple
            sample_hid_units = int(sample_feats * hid_units)
            pred_hid_units = int(predicate_feats * hid_units)
            join_hid_units = int(join_feats * hid_units)
            combined_size = sample_hid_units + pred_hid_units + join_hid_units
            combined_hid_units = int(combined_size * hid_units)
        else:
            ## takes less memory etc. when some have very few inp features
            sample_hid_units = int(min(hid_units, int(2*sample_feats)))
            pred_hid_units = int(min(hid_units, int(2*predicate_feats)))
            join_hid_units = int(min(hid_units, int(2*join_feats)))
            ## temporary to fix all input lengths being same
            # sample_hid_units = int(hid_units)
            # pred_hid_units = int(hid_units)
            # join_hid_units = int(hid_units)

            combined_size = sample_hid_units + pred_hid_units + join_hid_units

            if other_hid_units is None:
                combined_hid_units = hid_units
            else:
                combined_hid_units = other_hid_units


        if self.sample_feats != 0:
            sample_mlp1 = nn.Linear(sample_feats, sample_hid_units).to(device)
            self.sample_mlps.append(sample_mlp1)
            for i in range(0, self.num_hidden_layers-1):
                self.sample_mlps.append(nn.Linear(sample_hid_units,
                    sample_hid_units).to(device))


        if self.predicate_feats != 0:
            predicate_mlp1 = nn.Linear(predicate_feats, pred_hid_units).to(device)
            self.predicate_mlps.append(predicate_mlp1)
            for i in range(0, self.num_hidden_layers-1):
                self.predicate_mlps.append(nn.Linear(pred_hid_units,
                    pred_hid_units).to(device))

        if self.join_feats != 0:
            join_mlp1 = nn.Linear(join_feats, join_hid_units).to(device)
            self.join_mlps.append(join_mlp1)
            for i in range(0, self.num_hidden_layers-1):
                self.join_mlps.append(nn.Linear(join_hid_units,
                    join_hid_units).to(device))

        # Note: flow_feats is just used to concatenate global features, such as
        # pg_est at end of layer outputs

        comb_size = combined_size + flow_feats
        combined_hid_size = combined_hid_units + flow_feats

        self.out_mlp1 = nn.Linear(comb_size,
                combined_hid_size).to(device)

        # unless flow_feats is 0
        combined_hid_size += flow_feats
        self.out_mlp2 = nn.Linear(combined_hid_size, n_out).to(device)

    def forward(self, xbatch):
        '''
        #TODO: describe shapes
        '''
        start = time.time()

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
            samples = self.inp_drop_layer(samples)

            ## hardcoded layers
            # hid_sample = F.relu(self.sample_mlp1(samples))
            # hid_sample = self.hl_drop_layer(hid_sample)
            # if self.num_hidden_layers == 2:
                # hid_sample = F.relu(self.sample_mlp2(hid_sample))

            hid_sample = samples
            for i in range(0, self.num_hidden_layers):
                hid_sample = F.relu(self.sample_mlps[i](hid_sample))
                hid_sample = self.hl_drop_layer(hid_sample)

            sample_mask = sample_mask.to(device, non_blocking=True)
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

            # hid_predicate = F.relu(self.predicate_mlp1(predicates))
            # hid_predicate = self.hl_drop_layer(hid_predicate)
            # if self.num_hidden_layers == 2:
                # hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))

            hid_predicate = predicates
            for i in range(0, self.num_hidden_layers):
                hid_predicate = F.relu(self.predicate_mlps[i](hid_predicate))
                hid_predicate = self.hl_drop_layer(hid_predicate)

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

            # hid_join = F.relu(self.join_mlp1(joins))
            # hid_join = self.hl_drop_layer(hid_join)
            # if self.num_hidden_layers == 2:
                # hid_join = F.relu(self.join_mlp2(hid_join))

            hid_join = joins
            for i in range(0, self.num_hidden_layers):
                hid_join = F.relu(self.join_mlps[i](hid_join))
                hid_join = self.hl_drop_layer(hid_join)

            hid_join = hid_join * join_mask
            hid_join = torch.sum(hid_join, dim=1, keepdim=False)

            if torch.sum(join_mask) == 0:
                hid_join = torch.zeros(hid_join.shape).squeeze()
            else:
                join_norm = join_mask.sum(1, keepdim=False)
                hid_join = hid_join / join_norm
                hid_join = hid_join.squeeze()

            tocat.append(hid_join)

        if DEBUG_TIMES:
            inplayer_time = time.time()-start

        if self.flow_feats:
            flows = flows.to(device, non_blocking=True)
            flows = self.inp_drop_layer(flows)
            # hid_flow = F.relu(self.flow_mlp1(flows))
            # hid_flow = self.hl_drop_layer(hid_flow)
            # hid_flow = F.relu(self.flow_mlp2(hid_flow))
            tocat.append(flows)

        try:
            hid = torch.cat(tocat, 1)
        except Exception as e:
            print(e)
            print("forward pass torch.cat failed")
            pdb.set_trace()

        hid = self.combined_drop_layer(hid)

        hid = F.relu(self.out_mlp1(hid))
        if self.flow_feats:
            hid = torch.cat([hid, flows], 1)

        if self.use_sigmoid:
            out = torch.sigmoid(self.out_mlp2(hid))
        else:
            out = self.out_mlp2(hid)

        total_time = time.time()-start

        if DEBUG_TIMES:
            # print("Ratio total / input layer: ", round(inplayer_time / total_time, 6))
            self.total_fwd_time += total_time - inplayer_time

        return out

class SetConvFlow(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, flow_feats,
            hid_units, num_hidden_layers=2, n_out=1,
            dropouts=[0.0, 0.0, 0.0], use_sigmoid=True):
        super(SetConvFlow, self).__init__()
        self.use_sigmoid = use_sigmoid

        sample_feats = int(sample_feats)
        hid_units = int(hid_units)

        self.sample_feats = sample_feats
        self.predicate_feats = predicate_feats
        self.join_feats = join_feats
        self.flow_feats = flow_feats
        self.num_hidden_layers = num_hidden_layers

        num_layer1_blocks = 0

        self.inp_drop = dropouts[0]
        self.hl_drop = dropouts[1]
        self.combined_drop = dropouts[2]
        self.inp_drop_layer = nn.Dropout(p=self.inp_drop)
        self.hl_drop_layer = nn.Dropout(p=self.hl_drop)
        self.combined_drop_layer = nn.Dropout(p=self.combined_drop)

        # if self.num_hidden_layers > 2:
            # assert False, "need to implement"

        if self.sample_feats != 0:
            self.sample_mlp1 = nn.Linear(sample_feats, hid_units).to(device)
            if self.num_hidden_layers == 2:
                self.sample_mlp2 = nn.Linear(hid_units, hid_units).to(device)

            num_layer1_blocks += 1

        if self.predicate_feats != 0:
            self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units).to(device)
            if self.num_hidden_layers == 2:
                self.predicate_mlp2 = nn.Linear(hid_units, hid_units).to(device)

            num_layer1_blocks += 1

        if self.join_feats != 0:
            self.join_mlp1 = nn.Linear(join_feats, hid_units).to(device)
            if self.num_hidden_layers == 2:
                self.join_mlp2 = nn.Linear(hid_units, hid_units).to(device)
            num_layer1_blocks += 1

        if flow_feats != 0:
            self.flow_mlp1 = nn.Linear(flow_feats, hid_units).to(device)
            if self.num_hidden_layers == 2:
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

            if self.num_hidden_layers == 2:
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

            if self.num_hidden_layers == 2:
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

            if self.num_hidden_layers == 2:
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
            if self.num_hidden_layers == 2:
                hid_flow = F.relu(self.flow_mlp2(hid_flow))

            # if hid_flow.shape[0] == 1 and tocat[0].shape[0] != 1:
            hid_flow = hid_flow.squeeze()
            tocat.append(hid_flow)

        try:
            hid = torch.cat(tocat, 1)
            # if tocat[0].shape[0] == 1:
                # hid = torch.cat(tocat, 1)
            # else:
                # hid = torch.cat(tocat)
        except Exception as e:
            for tc in tocat:
                print(tc.shape)
            hid = torch.cat(tocat)
            # pdb.set_trace()

        hid = self.combined_drop_layer(hid)
        hid = F.relu(self.out_mlp1(hid))

        # if self.flow_feats:
            # hid = torch.cat([hid, flows], 1)

        if self.use_sigmoid:
            out = torch.sigmoid(self.out_mlp2(hid))
        else:
            out = self.out_mlp2(hid)
        return out

NUM_HEADS=4
class CardSetTransformer(nn.Module):
    def __init__(self, sample_feats, predicate_feats,
            join_feats,
            flow_feats,
            hid_units,
            num_hidden_layers=2, n_out=1,
            dropouts=[0.0, 0.0, 0.0], use_sigmoid=True):
        super(CardSetTransformer, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.sample_feats = sample_feats
        self.predicate_feats = predicate_feats
        self.join_feats = join_feats
        self.flow_feats = flow_feats
        self.num_hidden_layers = num_hidden_layers
        num_layer1_blocks = 0

        self.inp_drop = dropouts[0]
        self.hl_drop = dropouts[1]
        self.combined_drop = dropouts[2]
        self.inp_drop_layer = nn.Dropout(p=self.inp_drop)
        self.hl_drop_layer = nn.Dropout(p=self.hl_drop)
        self.combined_drop_layer = nn.Dropout(p=self.combined_drop)

        # self.sample_mlps = nn.ModuleList()
        # self.predicate_mlps = nn.ModuleList()
        # self.join_mlps = nn.ModuleList()

        if hid_units < 4:
            # gonna treat this as a multiple
            sample_hid_units = int(sample_feats * hid_units)
            pred_hid_units = int(predicate_feats * hid_units)
            join_hid_units = int(join_feats * hid_units)
            combined_size = sample_hid_units + pred_hid_units + join_hid_units
            combined_hid_units = int(combined_size * hid_units)
        else:
            # sample_hid_units = int(min(hid_units, int(2*sample_feats)))
            # pred_hid_units = int(min(hid_units, int(2*predicate_feats)))
            # join_hid_units = int(min(hid_units, int(2*join_feats)))
            ## need to make these multiples of NUM_HEADS
            # sample_hid_units = int(round(sample_hid_units / float(NUM_HEADS))*NUM_HEADS)
            # pred_hid_units = int(round(pred_hid_units / float(NUM_HEADS))*NUM_HEADS)
            # join_hid_units = int(round(join_hid_units / float(NUM_HEADS))*NUM_HEADS)

            sample_hid_units = hid_units
            pred_hid_units = hid_units
            join_hid_units = hid_units

            # sample_hid_units = hid_units
            # pred_hid_units = hid_units
            # join_hid_units = hid_units
            combined_size = sample_hid_units + pred_hid_units + join_hid_units
            combined_hid_units = int(hid_units)

        self.sample_mlps = nn.ModuleList()
        self.predicate_mlps = nn.ModuleList()

        if self.sample_feats != 0:
            sample_mlp1 = nn.Linear(sample_feats, sample_hid_units).to(device)
            self.sample_mlps.append(sample_mlp1)
            for i in range(0, self.num_hidden_layers-1):
                self.sample_mlps.append(nn.Linear(sample_hid_units,
                    sample_hid_units).to(device))

        if self.predicate_feats != 0:
            predicate_mlp1 = nn.Linear(predicate_feats, pred_hid_units).to(device)
            self.predicate_mlps.append(predicate_mlp1)
            for i in range(0, self.num_hidden_layers-1):
                self.predicate_mlps.append(nn.Linear(pred_hid_units,
                    pred_hid_units).to(device))

        # if self.sample_feats != 0:
            # self.sample_transformer = SetTransformer(sample_feats,
                                                  # 1,
                                                  # sample_hid_units,
                                                  # dim_hidden=sample_hid_units,
                                                  # num_heads=NUM_HEADS,
                                                  # ln=False).to(device)

        # if self.predicate_feats != 0:
            # print(predicate_feats)
            # self.predicate_transformer = SetTransformer(predicate_feats,
                                                        # 1,
                                                        # pred_hid_units,
                                                        # dim_hidden=pred_hid_units,
                                                        # num_heads=NUM_HEADS,
                                                        # ln=False).to(device)

        if self.join_feats != 0:
            self.join_transformer = SetTransformer(join_feats,
                                                        1,
                                                        join_hid_units,
                                                        dim_hidden=join_hid_units,
                                                        num_heads=NUM_HEADS,
                                                        ln=False).to(device)

        # Note: flow_feats is just used to concatenate global features, such as
        # pg_est at end of layer outputs

        comb_size = combined_size + flow_feats
        combined_hid_size = combined_hid_units + flow_feats

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
            ## transformer code
            # samples = samples.to(device, non_blocking=True)
            # samples = self.inp_drop_layer(samples)
            # hid_sample = self.sample_transformer(samples)
            # hid_sample = hid_sample.squeeze(1)
            # tocat.append(hid_sample)

            ## set convolution code
            samples = samples.to(device, non_blocking=True)
            samples = self.inp_drop_layer(samples)

            hid_sample = samples
            for i in range(0, self.num_hidden_layers):
                hid_sample = F.relu(self.sample_mlps[i](hid_sample))
                hid_sample = self.hl_drop_layer(hid_sample)

            sample_mask = sample_mask.to(device, non_blocking=True)
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
            ## transformer code
            # predicates = predicates.to(device, non_blocking=True)
            # predicates = self.inp_drop_layer(predicates)
            # hid_predicate = self.predicate_transformer(predicates)
            # hid_predicate = hid_predicate.squeeze(1)
            # tocat.append(hid_predicate)

            ## set conv code
            predicates = predicates.to(device, non_blocking=True)
            predicate_mask = predicate_mask.to(device, non_blocking=True)
            predicates = self.inp_drop_layer(predicates)

            hid_predicate = predicates
            for i in range(0, self.num_hidden_layers):
                hid_predicate = F.relu(self.predicate_mlps[i](hid_predicate))
                hid_predicate = self.hl_drop_layer(hid_predicate)

            hid_predicate = hid_predicate * predicate_mask
            hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
            predicate_norm = predicate_mask.sum(1, keepdim=False)
            hid_predicate = hid_predicate / predicate_norm
            hid_predicate = hid_predicate.squeeze()
            tocat.append(hid_predicate)

        if self.join_feats != 0:
            joins = joins.to(device, non_blocking=True)
            joins = self.inp_drop_layer(joins)
            hid_join = self.join_transformer(joins)
            hid_join = hid_join.squeeze(1)
            tocat.append(hid_join)

        if self.flow_feats:
            flows = flows.to(device, non_blocking=True)
            flows = self.inp_drop_layer(flows)
            tocat.append(flows)

        try:
            hid = torch.cat(tocat, 1)
        except Exception as e:
            print(e)
            print("forward pass torch.cat failed")
            pdb.set_trace()

        hid = self.combined_drop_layer(hid)

        # print(hid.shape)
        # pdb.set_trace()

        hid = F.relu(self.out_mlp1(hid))
        if self.flow_feats:
            hid = torch.cat([hid, flows], 1)

        if self.use_sigmoid:
            out = torch.sigmoid(self.out_mlp2(hid))
        else:
            out = self.out_mlp2(hid)
        return out

class SetConvCaptum(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, flow_feats,
            hid_units, num_hidden_layers=2, n_out=1,
            dropouts=[0.0, 0.0, 0.0]):
        super(SetConvCaptum, self).__init__()

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

        combined_hid_size = hid_units

        self.out_mlp1 = nn.Linear(hid_units * num_layer1_blocks,
                combined_hid_size).to(device)

        self.out_mlp2 = nn.Linear(combined_hid_size, n_out).to(device)

    # def forward(self, samples, predicates, joins, flows,
            # sample_mask, predicate_mask, join_mask):
    def forward(self, xbatch):
        '''
        #TODO: describe shapes
        '''
        # print("samples.shape: ", samples.shape)
        # print("sample_mask.shape: ", sample_mask.shape)
        # print(predicates.shape)
        samples = xbatch["table"]
        predicates = xbatch["pred"]
        joins = xbatch["join"]
        flows = xbatch["flow"]

        sample_mask = xbatch["tmask"]
        predicate_mask = xbatch["pmask"]
        join_mask = xbatch["jmask"]

        # samples = xbatch[0]
        # predicates = xbatch[1]
        # joins = xbatch[2]
        # flows = xbatch[3]

        # sample_mask = xbatch[4]
        # predicate_mask = xbatch[5]
        # join_mask = xbatch[6]

        # samples = xbatch["table"]
        # predicates = xbatch["pred"]
        # joins = xbatch["join"]
        # flows = xbatch["flow"]

        # sample_mask = xbatch["tmask"]
        # predicate_mask = xbatch["pmask"]
        # join_mask = xbatch["jmask"]

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
            self.inp_drop_layer(flows)
            hid_flow = F.relu(self.flow_mlp1(flows))
            hid_flow = self.hl_drop_layer(hid_flow)
            hid_flow = F.relu(self.flow_mlp2(hid_flow))
            tocat.append(hid_flow)

        # print(len(tocat))
        # print(tocat[0].shape)
        # print(tocat[1].shape)
        # print(tocat[2].shape)
        # print(hid.shape)

        hid = torch.cat(tocat, 1)
        hid = self.combined_drop_layer(hid)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        # print("out: ", out)
        return out

class SetConvNoFlow(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats,
            hid_units, num_hidden_layers=2, n_out=1,
            dropouts=[0.0, 0.0, 0.0]):
        super(SetConvNoFlow, self).__init__()

        self.sample_feats = sample_feats
        self.predicate_feats = predicate_feats
        self.join_feats = join_feats
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

        combined_hid_size = hid_units

        self.out_mlp1 = nn.Linear(hid_units * num_layer1_blocks,
                combined_hid_size).to(device)

        self.out_mlp2 = nn.Linear(combined_hid_size, n_out).to(device)

    def forward(self, samples, predicates, joins,
            sample_mask, predicate_mask, join_mask):
        '''
        #TODO: describe shapes
        '''
        # print("samples.shape: ", samples.shape)
        # print("sample_mask.shape: ", sample_mask.shape)
        # print(predicates.shape)
        # samples = xbatch["table"]
        # predicates = xbatch["pred"]
        # joins = xbatch["join"]
        # flows = xbatch["flow"]

        # sample_mask = xbatch["tmask"]
        # predicate_mask = xbatch["pmask"]
        # join_mask = xbatch["jmask"]

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


        # print(len(tocat))
        # print(tocat[0].shape)
        # print(tocat[1].shape)
        # print(tocat[2].shape)
        # print(hid.shape)

        hid = torch.cat(tocat, 1)
        hid = self.combined_drop_layer(hid)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        # print("out: ", out)
        return out
