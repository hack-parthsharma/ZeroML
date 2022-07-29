#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

"""
data :  feed inputs inot our network in batches.
        tools for iterating over data in batches.
"""
from typing import Iterator, NamedTuple
import numpy as np
from broNet.tensor import Tensor


Batch = NamedTuple('Batch', [('inputs', Tensor), ('targets', Tensor)])


class Data_Iterator:
    def __cal__(self, inputs: Tensor, targets: Tensor)-> Iterator[Batch]:
        raise NotImplementedError

class BatchIterator(Data_Iterator):
    def __init__(self, batch_size: int = 32, shuffle: bool=True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __cal__(self, inputs: Tensor, targets: Tensor)-> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
