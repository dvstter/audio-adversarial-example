#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T

class Filters:
  def __init__(self, height, width):
    # usage: matmal(data, _1)
    self._1 = T.eye(width)
    for y in range(width-1):
      self._1[:,y] -= self._1[:,y+1]

    # usage: matmal(_2, data)
    self._2 = T.eye(height)
    for x in range(height-1):
      self._2[x,:] -= self._2[x+1,:]

    # usage: matmal(data, _1)
    self._3 = T.eye(width)
    for y in range(width-2):
      self._3[:,y] -= 2*self._3[:,y+1] - self._3[:,y+2]

    # usage: matmal(_2, data)
    self._4 = T.eye(height)
    for x in range(height-2):
      self._4[x,:] -= 2*self._4[x+1,:] - self._4[x+2,:]

  def transform(self, input_data):
      batch, _, height, width = input_data.shape
      data = T.zeros([batch, 9, height, width])
      data[:, 0] = input_data[:, 0]
      data[:, 1] = T.matmul(input_data[:, 0], self._1)
      data[:, 2] = T.matmul(self._2, input_data[:, 0])
      data[:, 3] = T.matmul(input_data[:, 0].abs(), self._1)
      data[:, 4] = T.matmul(self._2, input_data[:, 0].abs())
      data[:, 5] = T.matmul(input_data[:, 0], self._3)
      data[:, 6] = T.matmul(self._4, input_data[:, 0])
      data[:, 7] = T.matmul(input_data[:, 0].abs(), self._3)
      data[:, 8] = T.matmul(self._4, input_data[:, 0].abs())
      return data