#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def norm(array):
  result = np.array(array, dtype=np.float)
  N = array.shape[0]
  temp = array.reshape(N, -1)
  _max = temp.max(1)
  for i in range(N):
    result[i] /= _max[i]
  return result.reshape(array.shape)

def multiple_gradients_value_guided_qmdct_modify(array, gradients, max_modifications, type='most', accuracy_direction='increase', neglect_sign=False, normalization=True):
  if isinstance(max_modifications, int):
    max_modifications = [max_modifications]
  if normalization:
    gradients = [norm(grads) for grads in gradients]
  grad = sum(gradients) / len(gradients)
  if type == 'most':
    return most_value_based_modify(array, grad, max_modifications, accuracy_direction)
  elif type == 'least':
    return least_value_based_modify(array, grad, max_modifications, accuracy_direction, neglect_sign)

def gradient_value_guided_qmdct_modify(array, gradient, max_modifications, type='most', accuracy_direction='increase', neglect_sign=False):
  if isinstance(max_modifications, int):
    max_modifications = [max_modifications]
  if type == 'most':
    return most_value_based_modify(array, gradient, max_modifications, accuracy_direction)
  elif type == 'least':
    return least_value_based_modify(array, gradient, max_modifications, accuracy_direction, neglect_sign)

def _preprocess(array, gradient, accuracy_direction='increase'):
  array = np.array(array, dtype=np.float)
  gradient = np.array(gradient)
  if accuracy_direction == 'increase':
    gradient = -gradient
  elif accuracy_direction != 'decrease':
    raise ValueError('gradient_guided_qmdct_modify param accuracy_direction should be increase or decrease only!')

  return array, gradient

def most_value_based_modify(array, gradient, max_modifications, accuracy_direction='increase'):
  array, gradient = _preprocess(array, gradient, accuracy_direction)

  index = np.logical_not(np.logical_and(np.logical_and(abs(array)>0, abs(array) <= 2), np.sign(array) != np.sign(gradient)))
  gradient[index] = .0
  num_array, height, width, _ = gradient.shape
  flattened_gradient = gradient.reshape(num_array, -1)
  flattened_index = np.flip(np.argsort(abs(flattened_gradient), 1), 1)

  result = {}
  for mm in max_modifications:
    temp_array = np.array(array.reshape(num_array, -1))
    for i in range(num_array):
      flip_index = flattened_index[i, :mm]
      temp_array[i, flip_index] = -temp_array[i, flip_index]

    result[mm] = temp_array.reshape(num_array, height, width, 1)
  return result

def least_value_based_modify(array, gradient, max_modifications, accuracy_direction='increase', neglect_sign=False):
  array, gradient = _preprocess(array, gradient, accuracy_direction)

  gradient[gradient == .0] = 100
  gradient[np.logical_or(array==0, abs(array) > 2)] = 100
  if neglect_sign is False:
    gradient[np.sign(gradient)==np.sign(array)] = 100
  num_array, height, width, _ = gradient.shape
  flattened_gradient = gradient.reshape(num_array, -1)
  flattened_index = np.argsort(abs(flattened_gradient), 1)

  result = {}
  for mm in max_modifications:
    temp_array = np.array(array.reshape(num_array, -1))
    for i in range(num_array):
      flip_index = flattened_index[i][:mm]
      temp_array[i, flip_index] = -temp_array[i, flip_index]
    result[mm] = temp_array.reshape(num_array, height, width, 1)
  return result

def gradient_sign_guided_qmdct_modify(array, gradient, accuracy_direction='increase', method='symbol'):
  array, gradient = _preprocess(array, gradient, accuracy_direction)

  if method not in ['linbit', 'symbol']:
    raise ValueError('gradient_guided_qmdct_modify param method should be linbit or symbol only!')

  num_array = array.shape[0]

  # 系数绝对值小于等于2且非0，异号
  if method=='symbol':
    index = np.logical_and(np.logical_and(abs(array)<=2, array!=0), np.sign(array)!=np.sign(gradient))
    array[index] = -array[index]
    modifications = index.reshape(num_array, -1).sum(axis=1)

  # 系数绝对值大于15，翻转LSB
  if method=='linbit':
    lsb = (abs(array)-15).clip(0)%2 # Note: 这个地方可能有问题，对于-值提取lsb可能有问题，但是暂时不管了

    index = np.logical_and(np.logical_and(abs(array)>15, gradient<0), lsb==1)
    array[index] -= 1
    modifications = index.reshape(num_array, -1).sum(axis=1)

    index = np.logical_and(np.logical_and(abs(array)>15, gradient>0), lsb==0)
    array[index] += 1
    modifications += index.reshape(num_array, -1).sum(axis=1)

  return array, modifications

def fgsm_qmdct_modify(array, gradient, epsilons=[0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0], max_modifications=[800, 1000, 2000, 4000, 8000, 10000, 20000], accuracy_direction='increase'):
  """
  just for verifying, not using

  :param array: ndarray
  :param gradient: ndarray
  :param epsilons: list of float
  :param max_modifications: list of int
  :param accuracy_direction: str, 'increase' or 'decrease'
  """

  array, gradient = _preprocess(array, gradient, accuracy_direction)

  num_array, height, width, _ = gradient.shape
  flattened_gradient = gradient.reshape(num_array, -1)
  flattened_array = array.reshape(num_array, -1)

  result = {}
  sorted_keys = []
  for type in ['most', 'least']:
    if type == 'most':
      flattened_index = np.argsort(abs(flattened_gradient), 1)
      flattened_index = np.flip(flattened_index, 1)
    elif type == 'least':
      leave_out_zero_gradient = np.array(flattened_gradient)
      leave_out_zero_gradient[leave_out_zero_gradient==.0] = 100
      flattened_index = np.argsort(abs(leave_out_zero_gradient), 1)
    for eps in epsilons:
      for mm in max_modifications:
        temp_index = flattened_index[:, :mm]
        temp_array = np.array(flattened_array)
        temp_key = f'{type}-{eps}-{mm}'
        for i in range(num_array):
          temp_array[i, temp_index[i]] += eps * np.sign(flattened_gradient[i, temp_index[i]])
        result[temp_key] = temp_array.reshape(num_array, height, width, 1)
        sorted_keys.append(temp_key)

  return result, sorted_keys

if __name__ == '__main__':
  pass