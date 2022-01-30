#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import utils
from tqdm import trange
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import config

def data_gradient(model, array, tensor_labels, clip=None, device=None):
  """
  calculate data gradient directly.

  :param model: RHFCN or WASDN model
  :param array: ndarry, [batch, height, width, 1]
  :param tensor_labels: torch.tensor, [batch], must be in the same device with array
  :param clip: float, clip too small gradient to 0.0
  :param device: str or None, if None, this will choose device automatically

  :return:
    probs: ndarray, [batch], original probability for corresponding label
    gradient: ndarray, [batch, height, widht, 1], same with array
  """

  tensor = utils.transform(array, device)
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

    return data_gradient(self._model, array, T.LongTensor(labels).to(self._device), device=self._device)

def _process_gradient(covers, grads):
  non_linbit_points = np.logical_or(abs(covers)>2, covers==0)
  non_used_grads_points = grads==.0
#  grads[np.logical_or(non_linbit_points, non_used_grads_points)] = 100 # modified first, for sorting
#  num_array, height, width, _ = grads.shape
#  flattened_gradient = grads.reshape(num_array, -1)
#  flattened_index = np.argsort(abs(flattened_gradient), 1)
#  for i in range(num_array):
#    flattened_gradient[i, flattened_index[i]] = np.arange(1, height*width+1)
#  gradient = flattened_gradient.reshape(num_array, height, width, 1)
#  # to save storage, but in next procedure, program should be aware of this problem
#  gradient[non_linbit_points] = 0
  # due to the effective matrix is only 200*450, so we should set the rest elements big enough to avoid usage in latter steps
#  gradient[non_used_grads_points] = 2000000
#  gradient = np.array(gradient, dtype=np.int)
  gradient = grads
  grads[np.logical_or(non_linbit_points, non_used_grads_points)] = .0
  return gradient

def _save_gradient(model, model_path, cover_path, gradient_path, label, batch_size=100):
  """
  save gradients for covers.

  :param model: str, 'rhfcn' or 'wasdn'
  :param model_path: str
  :param cover_path: str
  :param label: int, 0 or 1
  :param gradient_path: str, ended with '/'
  :param batch_size: int, batch size for calculation procedure
  """
  cover_files = utils.get_files_list(cover_path)
  gradient_files = [gradient_path + x.split('/')[-1] for x in cover_files]
  len_files = len(cover_files)
  device = utils.auto_select_device()

  print(f'read files from {cover_path}, len {len_files}, write gradient files to {gradient_path}.')

  gradient_calculator = DataGradientCalculator(model, model_path, device)
  cover_array = utils.text_read_batch(cover_files, progress=True)

  print('loading cover files over.')

  for i in trange(len_files // batch_size):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    _, gradient = gradient_calculator.gradient(cover_array[start_idx:end_idx], [label] * batch_size)
    gradient = _process_gradient(cover_array[start_idx:end_idx], gradient)
    utils.text_write_batch(gradient_files[start_idx:end_idx], gradient)

if __name__ == '__main__':
  cover_path, gradient_path = config.get_paths(320, stego=False, gradient=True)
  _save_gradient('wasdn', 'model_wasdn_local.pth', cover_path, gradient_path, 0)