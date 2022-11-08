#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils
import numpy as np

class MultiStegoPairDataset:
  def __init__(self, cover_path_or_files, stego_path_list_or_files_list, start_index=None, end_index=None, batch_size=None, device=None, progress=False, lazy=True):
    self._progress = progress
    self._data_loaded = not lazy
    if isinstance(cover_path_or_files, str):
      self._cover_files = utils.get_files_list(cover_path_or_files)[start_index:end_index]
    else:
      self._cover_files = cover_path_or_files
    if isinstance(stego_path_list_or_files_list[0], str):
      self._stego_files_list = [utils.get_files_list(path)[start_index:end_index] for path in stego_path_list_or_files_list]
    else:
      self._stego_files_list = stego_path_list_or_files_list
    assert all([len(x) == len(self._cover_files) for x in self._stego_files_list])
    self._len = len(self._cover_files)
    self._stego_types = len(self._stego_files_list)

    if self._data_loaded:
      self._cover_data = utils.text_read_batch(self._cover_files, progress=self._progress)
      self._stego_data_list = [utils.text_read_batch(stego_files, progress=self._progress) for stego_files in self._stego_files_list]

    self._batch_size = batch_size if batch_size else len(self._cover_data)

  def iterations(self):
    return self._len // self._batch_size

  def __len__(self):
    return self._len

  def __getitem__(self, idx):
    if self._data_loaded:
      cover = self._cover_data[idx:idx+1]
      stegos = [stego_data[idx:idx+1] for stego_data in self._stego_data_list]
    else:
      cover = utils.text_read_batch(self._cover_files[idx:idx+1], progress=False)
      stegos = [utils.text_read_batch(stego_files[idx:idx+1], progress=False) for stego_files in self._stego_files_list]
    data = np.concatenate([cover]+stegos)
    labels = [0] + [1]*self._stego_types

    return data, labels

  def __iter__(self):
    self._index = 0
    return self

  def __next__(self):
    if self._index >= self._len:
      raise StopIteration

    start = self._index
    self._index += self._batch_size
    end = self._index if self._index<=self._len else self._len
    actual_batch_size = end - start
    if self._data_loaded:
      cover = self._cover_data[start:end]
      stegos = [stego_data[start:end] for stego_data in self._stego_data_list]
    else:
      cover = utils.text_read_batch(self._cover_files[start:end], progress=False)
      stegos = [utils.text_read_batch(stego_files[start:end], progress=False) for stego_files in self._stego_files_list]

    data = np.concatenate([cover]+stegos)
    labels = [0] * actual_batch_size + [1] * self._stego_types * actual_batch_size

    return data, labels

class PairDataset(MultiStegoPairDataset):
  def __init__(self, cover_path_or_files, stego_path_or_files, start_index=None, end_index=None, batch_size=None, device=None, progress=False, lazy=True):
    super(PairDataset, self).__init__(cover_path_or_files, [stego_path_or_files], start_index, end_index, batch_size, device, progress, lazy)

class CrossValidationPairDataset(PairDataset):
  """
  this class has not been modified to accommodate to new feature - lazy loading, and this class is deprecated for now
  """
  def __init__(self, cover_path_or_files, stego_path_or_files, start_index=None, end_index=None, batch_size=10, to_tensor=False, device=None, progress=False, lazy=True, folds=10):
    super(CrossValidationPairDataset, self).__init__(cover_path_or_files, stego_path_or_files, start_index, end_index, batch_size, to_tensor, device, progress)
    self._folds = folds
    self._fold_size = self._len // self._folds
    if self._fold_size % self._batch_size != 0:
      raise ValueError('not supported!')
    self._valid_fold_indices = list(reversed(list(range(0, self._len, self._fold_size))))

  def iterations(self):
    return self._len * self._folds // self._batch_size

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