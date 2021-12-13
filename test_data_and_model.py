#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils
import numpy as np
import dataset
import config

def test_dataset():
  """
  测试dataset类实现的正确性
  """
  cover_path, stego_path = config.get_paths(birate=320)

  pds = dataset.PairDataset(cover_path, stego_path, end_index=100, folds=5, batch_size=10, to_tensor=dataset.ToTensor(utils.auto_select_device()))

  for t, D, l in pds:
    print(t, D.shape, len(l))



def test_model():
  """
  测试model在cover和stego上分别的准确率，以及区分的概率
  """
  pass


if __name__ == '__main__':
  test_dataset()
