#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import getpass
import os

def get_paths(birate='320', cover=True, stego=True, gradient=False):
  if not birate in ['128', '320', 128, 320]:
    raise ValueError('type must be 128 or 320')
  if (getpass.getuser()) == 'yanghanlin':
    cover_path = f'/Users/yanghanlin/Downloads/stego_analysis_data/Cover/{birate}/'
    stego_path = f'/Users/yanghanlin/Downloads/stego_analysis_data/Stego/ACS_B_{birate}_W_2_H_7_ER_10/'
    gradient_path = f'/Users/yanghanlin/Downloads/stego_analysis_data/Gradient/{birate}/'
  else:
    cover_path = f'/home/zhu/stego_analysis/Cover/{birate}/'
    stego_path = f'/home/zhu/stego_analysis/Stego/ACS_B_{birate}_W_2_H_7_ER_10/'
    gradient_path = f'/home/zhu/stego_analysis/Gradient/{birate}/'

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
