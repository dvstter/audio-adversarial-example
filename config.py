#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import getpass
import os

'''
def get_paths(birate='320', cover=True, stego=True, gradient=False):
  if not birate in ['128', '320', 128, 320, 'test']:
    raise ValueError('type must be 128, 320 or test')
  if (getpass.getuser()) == 'yanghanlin':
    cover_path = f'/Users/yanghanlin/Downloads/stego_analysis_data/Cover/{birate}/'
    stego_path = f'/Users/yanghanlin/Downloads/stego_analysis_data/Stego/ACS_B_{birate}_W_2_H_7_ER_10/'
    gradient_path = f'/Users/yanghanlin/Downloads/stego_analysis_data/Gradient/{birate}/'
  else:
    cover_path = f'/home/zhu/stego_analysis/Cover/{birate}/'
    stego_path = f'/home/zhu/stego_analysis/Stego/ACS_B_{birate}_W_2_H_7_ER_10/'
    gradient_path = f'/home/zhu/stego_analysis/Gradient/{birate}/'

  if birate == 'test':
    cover_path = '/home/zhu/stego_analysis/500_320/'
    stego_path = '/home/zhu/stego_analysis/500_320_stego/'
    gradient_path = '/home/zhu/stego_analysis/500_320_gradient'

  if gradient and not os.path.exists(gradient_path):
    os.makedirs(gradient_path)

  rets = []
  if cover:
    rets.append(cover_path)
  if stego:
    rets.append(stego_path)
  if gradient:
    rets.append(gradient_path)

  return rets
'''

def get_paths(train_or_test, stego_algos, embed_rates):
  """
  shortcuts to get paths

  @param train_or_test: str, 'train' or 'test'
  @param stego_algos: str, 'ags' or 'jed'
  @param embed_rates: list[int]
  """
  root_path = '/home/zhu/stego_analysis/'
  if train_or_test is 'train':
    root_path += 'train_data/'
  elif train_or_test is 'test':
    root_path += 'test_data/'

  cover_path = root_path + 'Cover/320'
  stego_paths = [root_path + 'Stego/' + stego_algos + '/w' + str(x) for x in embed_rates]

  return cover_path, stego_paths