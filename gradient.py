#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import utils
from tqdm import trange
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import config

def data_gradient(model, array, tensor_labels, clip=None):
  """
  calculate data gradient directly.

  :param model: RHFCN or WASDN model
  :param array: ndarry, [batch, height, width, 1]
  :param tensor_labels: torch.tensor, [batch], must be in the same device with array
  :param clip: float, clip too small gradient to 0.0

  :return:
    probs: ndarray, [batch], original probability for corresponding label
    gradient: ndarray, [batch, height, widht, 1], same with array
  """

  tensor = utils.transform(array)
  tensor.requires_grad = True
  output = model(tensor)
  loss = F.cross_entropy(output, tensor_labels)
  model.zero_grad()
  loss.backward()

  index = tensor_labels.unsqueeze(1)
  probs = F.softmax(output, dim=1).cpu()
  probs = T.gather(probs, 1, index).squeeze().detach().numpy()

  batch_size, height, width, _ = array.shape
  gradient = T.zeros([batch_size, height, width])
  gradient[:, :, :450] = tensor.grad.data[:, 0]
  gradient = gradient.cpu().numpy()
  gradient = gradient.reshape(array.shape)
  if clip:
    gradient[abs(gradient) < float(clip)] = 0.0
  return probs, gradient

def second_order_data_gradient(model, array, tensor_labels, clip=None):
  """
  calculate the first-order and second-order gradient directly.

  :param model: RHFCN or WASDN model
  :param array: ndarray, [batch, height, width, 1]
  :param tensor_labels: torch.tensor, [batch], must be in the same device with model
  :return:
    probs: ndarray, [batch], original probabilities for designated label
    gradient: ndarray, [batch, height, width, 1] which is totally same as param array
    second_gradient: ndarray, second-order gradient
  """
  probs, gradient = data_gradient(model, array, tensor_labels, clip)
  _, second_gradient = data_gradient(model, gradient, tensor_labels, clip)
  return probs, gradient, second_gradient

class DataGradientCalculator:
  def __init__(self, model, model_path, device=None):
    if not device:
      device = utils.auto_select_device()
    self._model = utils.load_model(model, model_path, device)
    self._criterion = nn.CrossEntropyLoss().to(device)
    self._device = device

  def gradient(self, array, labels):
    """
    calculate data gradient for array and label.

    :param array: ndarry, [batch, height, width, 1]
    :param labels: list/ndarry, dim == 1

    :return:
      gradient: ndarray, shape matches with array
    """

    return data_gradient(self._model, array, T.LongTensor(labels).to(self._device))

def _save_gradient(model='rhfcn', model_path='model_rhfcn.pth', batch_size=100):
#  cover_path, gradient_path = config.get_paths(birate=320, cover=True, stego=False, gradient=True)
  cover_path, gradient_path = '/home/zhu/stego_analysis/500_320/', '/home/zhu/stego_analysis/500_320_gradient/'
  cover_files = utils.get_files_list(cover_path)
  gradient_files = [gradient_path + x.split('/')[-1] for x in cover_files]
  len_files = len(cover_files)

  print(f'read files from {cover_path}, len {len_files}, write gradient files to {gradient_path}.')

  gradient_calculator = DataGradientCalculator(model, model_path)
  cover_array = utils.text_read_batch(cover_files, progress=True)

  print('loading cover files over.')

  # 查看梯度的各项数据，从而确定截断点
  def process_grads(_start_idx, _grads):
    n = _grads.shape[0]
    for i in range(n):
      _max, _min, _abs_min, _mean, _abs_mean = _grads[i].max(), _grads[i].min(), abs(_grads[i]).min(), _grads[i].sum() / (200*450), abs(_grads[i]).sum() / (200*450)
      print(f'{_start_idx + i} {_max} {_min} {_abs_min} {_mean} {_abs_mean}')
      return _max, _min, _abs_min, _mean, _abs_mean

#  _1, _2, _3, _4, _5 = [], [], [], [], []

  for i in trange(len_files // batch_size):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    array = cover_array[start_idx:end_idx]
    _, gradient = gradient_calculator.gradient(array, [1] * batch_size)
#    _max, _min, _abs_min, _mean, _abs_mean = process_grads(start_idx, gradient)
#    _1.append(_max), _2.append(_min), _3.append(_abs_min), _4.append(_mean), _5.append(_abs_mean)

#  print('total:')
#  print(max(_1), min(_2), min(_2), sum(_4)/(len_files // batch_size)), sum(_5)/(len_files // batch_size)

    _grads = np.array(np.sign(gradient), dtype=np.int)
    _grads[abs(gradient) < 1e-9] = 0 # 1e-9的截断点是经验值
    utils.text_write_batch(gradient_files[start_idx:end_idx], _grads)

if __name__ == '__main__':
  _save_gradient()