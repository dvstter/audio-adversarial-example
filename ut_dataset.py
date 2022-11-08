#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils
import numpy as np
from dataset import PairDataset as PD
from dataset import MultiStegoPairDataset as MultiPD
from dataset import CrossValidationPairDataset as CVPD
import config

def test_dataset():
  cover_path, stego_path = config.get_paths('test')
  _, stego_path2 = config.get_paths('test', embed_rate='4')
  _, stego_path3 = config.get_paths('test', embed_rate='8')

  cover_path2, stego_path4 = config.get_paths('train', stego_algos='ags-small', embed_rate='20000')
  _, stego_path5 = config.get_paths('train', stego_algos='ags-big', embed_rate='20000')
  cover_files = utils.get_files_list(cover_path2)

  device = utils.auto_select_device()

  test_set = [PD(cover_path, stego_path, end_index=100, batch_size=40, device=device, lazy=False),
              PD(cover_path, stego_path, end_index=100, batch_size=40, device=device),
              MultiPD(cover_path, [stego_path2, stego_path3], end_index=100, batch_size=30, lazy=False),
              MultiPD(cover_path, [stego_path2, stego_path3], end_index=100, batch_size=30),
              MultiPD(cover_path, [stego_path2, stego_path3], end_index=100, lazy=False),
              MultiPD(cover_files, [stego_path4, stego_path5], batch_size=100)]
  for idx, pds in enumerate(test_set):
    print(f'-----------------{idx}th--------------------')
    print(f'test __getitem__ method:')
    data, labels = pds[5]
    print(data.shape, labels)
    print(f'test __next__ method:')
    for D, l in pds:
      print(D.shape, len(l))

if __name__ == '__main__':
  test_dataset()
