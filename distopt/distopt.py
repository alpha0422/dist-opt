#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import _distopt

class DictWatch(dict):
    def __init__(self, adict, callback):
        super(DictWatch, self).__init__()
        self.update(adict)
        self.callback = callback

    def __setitem__(self, key, val):
        self.callback(key, val)
        dict.__setitem__(self, key, val)

class DistributedFusedAdam(torch.optim.Adam):
    """
    Wrap distributed optimizer with LR scheduler support.
    """
    def __init__(self, *args, **kwargs):
        config = {'lr':0.001, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        for key, val in config.items():
            if key in kwargs:
                config[key] = kwargs[key]
        self._params = [torch.nn.Parameter(torch.zeros(1))]
        super(DistributedFusedAdam, self).__init__(self._params, **config)

        self._dfa = _distopt.DistributedFusedAdam(*args, **kwargs)

        # Change LR for distributed optimizer once updated
        def hook(item, value):
            if item == 'lr':
                self._dfa.lr(value)

        self.param_groups[0] = DictWatch(self.param_groups[0], hook)

    def __getattr__(self, key):
        # Forward all other distributed optimizer method here
        # PyTorch LR scheduler require a weak reference to bound method
        # Refer to torch/optim/lr_scheduler.py:56
        return getattr(self._dfa, key)

    def step(self, *args, **kwargs):
        # Forward step to distributed optimizer
        return self._dfa.step(*args, **kwargs)

