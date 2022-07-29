#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np

from broNet.train import train
from broNet.nn import NeuralNet
from broNet.layers import Linear, Tanh


inputs = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])
targets = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0]
])
net = NeuralNet([
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)






