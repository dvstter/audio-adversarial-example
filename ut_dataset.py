#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils
import numpy as np
from dataset import PairDataset as PD
from dataset import CrossValidationPairDataset as CVPD
import dataset
import config

def test_dataset():
  """
  测试dataset类实现的正确性
  """
  cover_path, stego_path = config.get_paths('train')

  cvpds = CVPD(cover_path, stego_path, end_index=100, folds=5, batch_size=20, to_tensor=True, device=utils.auto_select_device())
  for t, D, l in cvpds:
    print(t, D.shape, len(l))

  pds = PD(cover_path, stego_path, end_index=100, batch_size=10, to_tensor=True, device=None)
  for D, l in pds:
    print(D.shape, len(l))

if __name__ == '__main__':
  test_dataset()
