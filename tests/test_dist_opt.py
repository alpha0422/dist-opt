#!/usr/bin/env python

import torch
import unittest
from distopt import DistributedFusedAdam


class DistributedFusedAdamTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)

    def test_single(self):
        model = torch.nn.LSTM(1024, 1024).cuda().half()
        params = list(model.parameters())
        opt = DistributedFusedAdam(params)
        opt.step()

if __name__ == '__main__':
    unittest.main()
