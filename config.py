#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import getpass
import os

def get_multi_stego_paths(train_or_test, stego_algos, embed_rates, root_path='/home/zhu/stego_analysis/'):
  """
  shortcuts to get paths

  :param train_or_test: str, 'train' or 'test'
  :param stego_algos: list[str], 'ags' or 'jed'
  :param embed_rates: list[int]
  :param root_path: str, just like '/home/zhu/stego_analysis/' must be ended with '/'
  """

  assert len(stego_algos) == len(embed_rates) or len(stego_algos) == 1
  if train_or_test == 'train':
    root_path += 'train_data/'
  elif train_or_test == 'test':
    root_path += 'test_data/'

  cover_path = root_path + 'Cover/'

  if len(stego_algos) == 1:
    stego_paths = [f'{root_path}Stego/{stego_algos[0]}/w{rate}' for rate in embed_rates]
  else:
    stego_paths = [f'{root_path}Stego/{stego_algos[i]}/w{embed_rates[i]}' for i in range(len(stego_algos))]

  return cover_path, stego_paths

def get_paths(train_or_test, stego_algo='jed', embed_rate='2'):
  cover_path, stego_paths = get_multi_stego_paths(train_or_test, [stego_algo], [embed_rate])
  return cover_path, stego_paths[0]
