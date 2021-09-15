import torch
import numpy as np
from query_representation.utils import *
from .dataset import QueryDataset, pad_sets, to_variable
from .nets import *
from .algs import *
# import cardinality_estimation.algs as algs

from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_

def mscn_collate_fn(data):
    '''
    TODO: faster impl.
    '''
    start = time.time()
    alltabs = []
    allpreds = []
    alljoins = []

    flows = []
    ys = []
    infos = []

    maxtabs = 0
    maxpreds = 0
    maxjoins = 0

    for d in data:
        alltabs.append(d[0]["table"])
        if len(alltabs[-1]) > maxtabs:
            maxtabs = len(alltabs[-1])

        allpreds.append(d[0]["pred"])
        if len(allpreds[-1]) > maxpreds:
            maxpreds = len(allpreds[-1])

        alljoins.append(d[0]["join"])
        if len(alljoins[-1]) > maxjoins:
            maxjoins = len(alljoins[-1])

        flows.append(d[0]["flow"])
        ys.append(d[1])
        infos.append(d[2])

    tf,pf,jf,tm,pm,jm = pad_sets(alltabs, allpreds,
            alljoins, maxtabs,maxpreds,maxjoins)

    flows = to_variable(flows, requires_grad=False).float()
    ys = to_variable(ys, requires_grad=False).float()
    data = {}
    data["table"] = tf
    data["pred"] = pf
    data["join"] = jf
    data["flow"] = flows
    data["tmask"] = tm
    data["pmask"] = pm
    data["jmask"] = jm

    return data,ys,infos

class MSCN(CardinalityEstimationAlg):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        for k, val in kwargs.items():
            self.__setattr__(k, val)

        # when estimates are log-normalized, then optimizing for mse is
        # basically equivalent to optimizing for q-error
        if self.loss_func_name == "qloss":
            self.loss_func = qloss_torch
        elif self.loss_func_name == "mse":
            self.loss_func = torch.nn.MSELoss(reduction="none")
        else:
            assert False

        if self.load_padded_mscn_feats:
            self.collate_fn = None
        else:
            self.collate_fn = mscn_collate_fn

    def init_dataset(self, samples):
        ds = QueryDataset(samples, self.featurizer, False,
                load_padded_mscn_feats=self.load_padded_mscn_feats)
        return ds

    def init_net(self, sample):

        if len(sample[0]["table"]) == 0:
            sfeats = 0
        else:
            sfeats = len(sample[0]["table"][0])

        if len(sample[0]["pred"]) == 0:
            pfeats = 0
        else:
            pfeats = len(sample[0]["pred"][0])

        if len(sample[0]["join"]) == 0:
            jfeats = 0
        else:
            jfeats = len(sample[0]["join"][0])

        net = SetConv(sfeats,
                pfeats, jfeats,
                len(sample[0]["flow"]),
                self.hidden_layer_size)
        print(net)

        if self.optimizer_name == "ams":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=True, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=False, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=False, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(net.parameters(),
                    lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            assert False

        return net, optimizer

    def train_one_epoch(self):
        for idx, (xbatch,ybatch,info) \
                    in enumerate(self.trainloader):
            ybatch = ybatch.to(device, non_blocking=True)
            pred = self.net(xbatch["table"],xbatch["pred"],xbatch["join"],
                    xbatch["flow"],xbatch["tmask"],xbatch["pmask"],
                    xbatch["jmask"]).squeeze(1)
            assert pred.shape == ybatch.shape

            # print(self.training_samples[0]["name"])
            # print(tbatch.shape, pbatch.shape, jbatch.shape)
            # print(torch.mean(tbatch), torch.mean(pbatch), torch.mean(jbatch))
            # print(torch.sum(tbatch), torch.sum(pbatch), torch.sum(jbatch))
            # pdb.set_trace()

            losses = self.loss_func(pred, ybatch)
            loss = losses.sum() / len(losses)

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(self.net.parameters(), self.clip_gradient)
            self.optimizer.step()

    def train(self, training_samples, **kwargs):
        assert isinstance(training_samples[0], dict)
        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples

        self.trainds = self.init_dataset(training_samples)
        self.trainloader = data.DataLoader(self.trainds,
                batch_size=self.mb_size, shuffle=True,
                collate_fn=self.collate_fn)

        # TODO: initialize self.num_features
        self.net, self.optimizer = self.init_net(self.trainds[0])

        model_size = self.num_parameters()
        print("""training samples: {}, model size: {},
        hidden_layer_size: {}""".\
                format(len(self.trainds), model_size,
                    self.hidden_layer_size))

        for self.epoch in range(0,self.max_epochs):
            # TODO: add periodic evaluation here
            start = time.time()
            self.train_one_epoch()
            print("train epoch took: ", time.time()-start)
            # pdb.set_trace()

    def num_parameters(self):
        def _calc_size(net):
            model_parameters = net.parameters()
            params = sum([np.prod(p.size()) for p in model_parameters])
            # convert to MB
            return params*4 / 1e6

        num_params = _calc_size(self.net)
        return num_params

    def _eval_ds(self, ds):
        # torch.set_num_threads(1)
        torch.set_grad_enabled(False)

        # important to not shuffle the data so correct order preserved!
        loader = data.DataLoader(ds,
                batch_size=5000, shuffle=False,
                collate_fn=self.collate_fn)

        allpreds = []

        for (xbatch,ybatch,info) in loader:
            ybatch = ybatch.to(device, non_blocking=True)
            pred = self.net(xbatch["table"],xbatch["pred"],xbatch["join"],
                    xbatch["flow"],xbatch["tmask"],xbatch["pmask"],
                    xbatch["jmask"]).squeeze(1)
            allpreds.append(pred)

        preds = torch.cat(allpreds).detach().cpu().numpy()
        torch.set_grad_enabled(True)

        return preds

    def test(self, test_samples, **kwargs):
        '''
        '''
        testds = self.init_dataset(test_samples)
        preds = self._eval_ds(testds)

        return format_model_test_output(preds, test_samples, self.featurizer)
