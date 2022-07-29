#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

"""
Train : a function that can train a neural network

"""

from broNet.tensor import Tensor
from broNet.nn import NeuralNet
from broNet.loss import Loss, MSE
from broNet.optim import Optimizer,SGD
from broNet.data import Data_Iterator, BatchIterator


def train(net: NeuralNet, 
          inputs: Tensor, 
          targets: Tensor, 
          num_epochs: int= 5000, 
          iterator: Data_Iterator= BatchIterator(),
          loss: Loss =MSE(),
          optimizer: Optimizer= SGD()
          )-> None:

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)

