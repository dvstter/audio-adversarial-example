#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rn
import torch as T

import gradient
import utils
import config

def test_gradient_heterogeneous():
  path, _ = config.get_paths(birate=320)
  files = utils.get_files_list(path)[:50]
  array = utils.text_read_batch(files, progress=True)
  device = utils.auto_select_device()
  model = utils.load_model('rhfcn', 'model_rhfcn.pth')
  _, grads1 = gradient.data_gradient(model, array, T.LongTensor([0]*50).to(device))
  _, grads2 = gradient.data_gradient(model, array, T.LongTensor([1]*50).to(device))

  grads1 = -grads1.reshape(50, -1)
  grads2 = grads2.reshape(50, -1)

  indices = rn.random_integers(0, grads1.shape[1], 20)

  for i in range(50):
    print(i)
    print(grads1[i][indices])
    print(grads2[i][indices])


if __name__ == '__main__':
  test_gradient_heterogeneous()

