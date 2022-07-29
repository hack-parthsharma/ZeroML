#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

"""
Layers :
    Linear: o = inputs @ w + b
"""
from typing import Dict,Callable
import numpy as np
from broNet.tensor import Tensor 


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size: int, output_size:int) -> None:
        super().__init__()
        self.params['w']= np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)

    def forward(self,inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']

    def backward(self, grad: Tensor) -> Tensor:
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad @ self.params['w'].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    def __init__(self,f:F, f_prime: F) -> None:
        super().__init__()
        self.f =f
        self.f_prime = f_prime
    
    def forward(self,inputs: Tensor)->Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self,grad:Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad 

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x:Tensor) -> Tensor:
    y = tanh(x)
    return 1- y ** 2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)
        











