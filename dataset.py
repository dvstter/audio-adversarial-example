#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils
import numpy as np
import torch as T

class ToTensor:
  def __init__(self, device):
    self.device = device

class PairDataset:
  def __init__(self, cover_path, stego_path, start_index=None, end_index=None, folds=10, batch_size=10, to_tensor=None):
    self._batch_size = batch_size
    cover_files, stego_files = utils.get_files_list(cover_path)[start_index:end_index], utils.get_files_list(stego_path)[start_index:end_index]
    self._len = len(cover_files)
    if len(cover_files) != len(stego_files):
      raise ValueError('cover files should be paired with stego files.')
    self._cover_data = utils.text_read_batch(cover_files, progress=True)
    self._stego_data = utils.text_read_batch(stego_files, progress=True)
    if isinstance(to_tensor, ToTensor):
      self._cover_data = utils.transform(self._cover_data, to_tensor.device)
      self._stego_data = utils.transform(self._stego_data, to_tensor.device)
      self._cat_method = T.cat
    else:
      self._cat_method = np.concatenate
    self._folds = folds
    self._fold_size = self._len // self._folds
    if self._fold_size % self._batch_size != 0:
      raise ValueError('not supported!')
    self._valid_fold_indices = list(reversed(list(range(0, self._len, self._fold_size))))
    self._batch_size = batch_size

  def iterations(self):
    return self._len * self._folds // self._batch_size

  def __len__(self):
    return self._len

  def __getitem__(self, idx):
    data = self._cat_method([self._cover_data[idx], self._stego_data[idx]])
    labels = [0,1]

    return data, labels

  def __iter__(self):
    self._index = 0
    self._current_fold = 0
    return self

  def __next__(self):
    if self._current_fold >= self._folds:
      raise StopIteration

    else:
      tag = 'T'
      valid_fold_index = self._valid_fold_indices[self._current_fold]
      if valid_fold_index <= self._index < valid_fold_index + self._fold_size:
        tag = 'V'

      data = self._cat_method([self._cover_data[self._index:self._index+self._batch_size],
                             self._stego_data[self._index:self._index+self._batch_size]])
      labels = [0]*self._batch_size + [1]*self._batch_size

      self._index += self._batch_size
      if self._index >= self._len:
        self._current_fold += 1
        self._index = 0
      return tag, data, labels