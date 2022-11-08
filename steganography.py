#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as T
import utils, tqdm, gradient

def norm(array):
  result = np.array(array, dtype=np.float)
  N = array.shape[0]
  temp = array.reshape(N, -1)
  _max = temp.max(1)
  for i in range(N):
    result[i] /= _max[i]
  return result.reshape(array.shape)

def assert_modify_correct(cover, stego, domain='small'):
  """
  verify cover is correctly converted to stego

  :param cover: ndarray, [batch, height, width, 1]
  :param stego: ndarry, [batch, height, width, 1]
  :param domain: str, 'big' or 'small'
  """
  if domain == 'small':
    assert (abs(cover[cover != stego]) > 2).reshape(-1).sum() == 0
  elif domain == 'big':
    assert (abs(cover[cover != stego]) <= 2).reshape(-1).sum() == 0
  temp = np.array(cover)
  temp[:, :200, :450] = stego[:, :200, :450]
  assert (temp!=stego).reshape(-1).sum() == 0

def multiple_gradients_value_guided_qmdct_modify(array, gradients, max_modifications, type='most', domain='small', accuracy_direction='increase', neglect_sign=False, normalization=True, gradients_sum_up_guide=False):
  """
  use gradient value to guide qmdct modify with multiple max_modifications

  :param array: ndarray, [batch, height, width, 1]
  :param gradients: ndarray, same size with array
  :param max_modifications: list[int, ...]
  :param type: str, 'most' or 'least'
  :param domain: str, 'small' or 'big'
  :param accuracy_direction: str, 'increase' or 'decrease'
  :param neglect_sign: bool, neglect gradients' sign or not
  :param normalization: bool, normalize gradients according to the batch dimension
  :param gradients_sum_up_guide: bool, default False, sum up all gradients to generate one generalized gradient to guide all the procedures

  :return:
    result: dict, {max_modification: ndarray}
  """
  if isinstance(max_modifications, int):
    max_modifications = [max_modifications]
  if normalization:
    gradients = [norm(grads) for grads in gradients]
  if gradients_sum_up_guide:
    grad = sum(gradients) / len(gradients)
    gradients[:,] = grad

  assert (type in ['most', 'least']) and (domain in ['small', 'big'] and (accuracy_direction in ['increase', 'decrease']))

  if type == 'most':
    return most_value_based_modify(array, gradients, domain, max_modifications, accuracy_direction)
  else:
    return least_value_based_modify(array, gradients, domain, max_modifications, accuracy_direction, neglect_sign)

def _preprocess(array, gradient, accuracy_direction='increase'):
  array = np.array(array, dtype=np.float)
  gradient = np.array(gradient)
  if accuracy_direction == 'increase':
    gradient = -gradient
  elif accuracy_direction != 'decrease':
    raise ValueError('gradient_guided_qmdct_modify param accuracy_direction should be increase or decrease only!')

  return array, gradient

def most_value_based_modify(array, gradient, domain, max_modifications, accuracy_direction='increase'):
  """
  notice that modify big value domain is not supported now
  """
  array, gradient = _preprocess(array, gradient, accuracy_direction)

  non_modify_region = np.logical_or(array==0, abs(array) > 2) if domain == 'small' else abs(array)<=2
  # most value based modify must consider the sign direction
  non_modify_region = np.logical_or(non_modify_region, np.sign(gradient) == np.sign(array))
  gradient[non_modify_region] = .0
  batch, height, width, _ = gradient.shape
  flattened_gradient = gradient.reshape(batch, -1)
  flattened_index = np.flip(np.argsort(abs(flattened_gradient), 1), 1)

  result = {}
  for mm in max_modifications:
    temp_array = np.array(array.reshape(batch, -1))
    for i in range(batch):
      flip_index = flattened_index[i, :mm]
      temp_array[i, flip_index] = -temp_array[i, flip_index]

    temp_array = temp_array.reshape(batch, height, width, 1)
    temp_array[non_modify_region] = array[non_modify_region]
    result[mm] = np.array(array, dtype=int)
    result[mm][:, :200, :450] = temp_array[:, :200, :450]
  return result

def least_value_based_modify(array, gradient, domain, max_modifications, accuracy_direction='increase', neglect_sign=False):
  array, gradient = _preprocess(array, gradient, accuracy_direction)

  gradient[gradient == .0] = 100 # avoid modify the region outside the 200*450 matrix
  non_modify_region = np.logical_or(array==0, abs(array) > 2) if domain == 'small' else abs(array)<=2
  gradient[non_modify_region] = 100
  if not neglect_sign:
    gradient[np.sign(gradient)==np.sign(array)] = 100
  batch, height, width, _ = gradient.shape
  flattened_gradient = gradient.reshape(batch, -1)
  flattened_index = np.argsort(abs(flattened_gradient), 1)

  result = {}
  for mm in max_modifications:
    temp_array = np.array(array.reshape(batch, -1))
    for i in range(batch):
      flip_index = flattened_index[i][:mm]
      temp_array[i, flip_index] = -temp_array[i, flip_index]

    temp_array = temp_array.reshape(batch, height, width, 1)
    temp_array[non_modify_region] = array[non_modify_region]
    result[mm] = np.array(array, dtype=int)
    result[mm][:, :200, :450] = temp_array[:, :200, :450]
  return result

def gradient_sign_guided_qmdct_modify(array, gradient, accuracy_direction='increase', method='symbol'):
  """
  there are some problem need to be fixed, so this method should not be used any longer
  """
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

if __name__ == '__main__':
  pass