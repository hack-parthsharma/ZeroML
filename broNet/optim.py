#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

"""
optimizer : to adjust the parameters of our network 
            based on the gradients 
            computed during propogation

"""
from broNet.nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad











