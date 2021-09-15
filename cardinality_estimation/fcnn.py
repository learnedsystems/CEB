import torch
import numpy as np
from query_representation.utils import *
from .dataset import QueryDataset, pad_sets, to_variable
from .nets import *
from .algs import *
# import cardinality_estimation.algs as algs

from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_

class FCNN(CardinalityEstimationAlg):
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

        self.collate_fn = None

    def init_dataset(self, samples):
        ds = QueryDataset(samples, self.featurizer, False)
        return ds

    def init_net(self, sample):
        net = SimpleRegression(self.num_features, 1,
                self.num_hidden_layers, self.hidden_layer_size)
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

        for idx, (xbatch, ybatch,info) in enumerate(self.trainloader):

            ybatch = ybatch.to(device, non_blocking=True)
            xbatch = xbatch.to(device, non_blocking=True)

            pred = self.net(xbatch).squeeze(1)
            assert pred.shape == ybatch.shape

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

        self.num_features = len(self.trainds[0][0])
        # TODO: initialize self.num_features
        self.net, self.optimizer = self.init_net(self.trainds[0])

        model_size = self.num_parameters()
        print("""training samples: {}, feature length: {}, model size: {},
        hidden_layer_size: {}""".\
                format(len(self.trainds), self.num_features, model_size,
                    self.hidden_layer_size))

        for self.epoch in range(0,self.max_epochs):
            # TODO: add periodic evaluation here
            start = time.time()
            self.train_one_epoch()

    def num_parameters(self):
        def _calc_size(net):
            model_parameters = net.parameters()
            params = sum([np.prod(p.size()) for p in model_parameters])
            # convert to MB
            return params*4 / 1e6

        num_params = _calc_size(self.net)
        return num_params

    def _eval_ds(self, ds):
        torch.set_grad_enabled(False)
        loader = data.DataLoader(ds,
                batch_size=5000, shuffle=False,
                collate_fn=self.collate_fn)
        allpreds = []

        for xbatch, ybatch,info in loader:
            ybatch = ybatch.to(device, non_blocking=True)
            xbatch = xbatch.to(device, non_blocking=True)
            pred = self.net(xbatch).squeeze(1)
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
