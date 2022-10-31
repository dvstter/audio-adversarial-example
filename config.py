#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import getpass
import os

def get_paths(train_or_test, stego_algos='jed', embed_rate='2'):
  """
  shortcuts to get paths

  @param train_or_test: str, 'train' or 'test'
  @param stego_algos: str, 'ags' or 'jed'
  @param embed_rates: list[int]
  """
  root_path = '/home/zhu/stego_analysis/'
  if train_or_test == 'train':
    root_path += 'train_data/'
  elif train_or_test == 'test':
    root_path += 'test_data/'

  cover_path = root_path + 'Cover/'
  stego_path = f'{root_path}Stego/{stego_algos}/w{embed_rate}'

  return cover_path, stego_path