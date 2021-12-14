#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
import tqdm

import gradient
import utils as utils
import config as config
import steganography as steganography

def _load_everything(model, model_path):
  cover_path = '/home/zhu/stego_analysis/500_320'
  cover_files = utils.get_files_list(cover_path)
  batch_size = 100
  len_files = len(cover_files)
  batches = len_files // batch_size
  cover_array = utils.text_read_batch(cover_files, progress=True)
  device = utils.auto_select_device()
  model = utils.load_model(model, model_path, device)

  return model, device, cover_array, batches, batch_size, len_files

def generate_and_test(model, criterion, model_path=None, criterion_path=None,
                      max_modifications=[100, 200, 500, 1000, 2000, 4000, 8000, 10000, 12000, 20000],
                      gradient_label=0, modified_label=0, accuracy_direction='increase',
                      save_path='qmdct_gradient_value_based_result.csv', gradient_prefer_type='most', neglect_sign=False):
  model_path = model_path if model_path else f'model_{model}.pth'
  criterion_path = criterion_path if criterion_path else f'model_{criterion}.pth'
  model, device, cover_array, batches, batch_size, len_files = _load_everything(model, model_path)
  if model_path == criterion_path:
    criterion = model
  else:
    criterion = utils.load_model(criterion, criterion_path, device).eval()

  result = {'original': [], 'modified': {mm: [] for mm in max_modifications}, 'modifications': {mm: [] for mm in max_modifications}}

  for i in tqdm.trange(batches):
    start = i * batch_size
    end = start + batch_size

    original_probs, grads = gradient.data_gradient(model, cover_array[start:end], T.LongTensor([gradient_label]*batch_size).to(device))
    modified_arrays = steganography.gradient_value_guided_qmdct_modify(cover_array[start:end], grads,
                                                                       max_modifications=max_modifications, type=gradient_prefer_type,
                                                                       accuracy_direction=accuracy_direction, neglect_sign=neglect_sign)
    for mm in modified_arrays.keys():
      result['modified'][mm] += list(criterion.get_probabilities(utils.transform(modified_arrays[mm], device), [modified_label]*batch_size))
#      result['modifications'][mm] += list((modified_arrays[mm].reshape(batch_size, -1)!=cover_array[start:end].reshape(batch_size, -1)).sum(axis=1))
    result['original'] += list(original_probs)

  with open(save_path, 'wt') as f:
    f.write('original_prob,'+','.join(['d'+str(mm) for mm in max_modifications])+'\n')
    for i in range(len_files):
      s = str(result['original'][i]) + ',' + ','.join([str(result['modified'][mm][i]) for mm in max_modifications]) + '\n'
      f.write(s)

  print('saved results!')

def test_gradient_value_guided_qmdct_modify(model='rhfcn', model_path='model_rhfcn.pth',
                                            max_modifications=[100, 200, 500, 1000, 2000, 4000, 8000, 10000, 12000, 20000],
                                            gradient_label=0, modified_label=0, accuracy_direction='increase',
                                            save_path='qmdct_gradient_value_based_result.csv', gradient_prefer_type='most', neglect_sign=False):
  generate_and_test(model, model, model_path=model_path, criterion_path=model_path, max_modifications=max_modifications,
                    gradient_label=gradient_label, modified_label=modified_label, accuracy_direction=accuracy_direction,
                    save_path=save_path, gradient_prefer_type=gradient_prefer_type, neglect_sign=neglect_sign)

def test_two_models(generator_model='wasdn', criterion_model='rhfcn', save_path='blackbox.csv'):
  generate_and_test(generator_model, criterion_model, gradient_prefer_type='least', save_path=save_path)

def test_gradient_sign_guided_qmdct_modify(model='rhfcn', model_path='model_rhfcn.pth'):
  model, device, cover_array, batches, batch_size, len_files = _load_everything(model, model_path)
  result = {'ori': [], 'mod': [], 'modifications': []}
  for i in tqdm.trange(batches):
    start = i * batch_size
    end = start + batch_size

    original_probs, grads = gradient.data_gradient(model, cover_array[start:end], T.LongTensor([1]*batch_size).to(device))
    modified_array, modifications = steganography.gradient_sign_guided_qmdct_modify(cover_array[start:end], grads, accuracy_direction='decrease')
    modified_probs = model.get_probabilities(utils.transform(modified_array, device), [0]*batch_size)

    result['ori'] += list(original_probs)
    result['mod'] += list(modified_probs)
    result['modifications'] += list(modifications)

  save_path = 'qmdct_sign_based_result.csv'
  with open(save_path, 'wt') as f:
    f.write('original,modified,modifications\n')
    for i in range(len_files):
      f.write('{},{},{}\n'.format(result['ori'][i], result['mod'][i], result['modifications'][i]))
  print(f'file saved to {save_path}')

def test_fgsm_qmdct_modify(model='rhfcn', model_path='model_rhfcn.pth'):
  model, device, cover_array, batches, batch_size, len_files = _load_everything(model, model_path)

  # test modify stego
  cover_array = utils.text_read_batch(utils.get_files_list(config.get_paths(birate=320)[1])[:500])


  result = {'original': []}

  for i in tqdm.trange(batches):
    start = i * batch_size
    end = start + batch_size

    original_probs, grads = gradient.data_gradient(model, cover_array[start:end], T.LongTensor([1] * batch_size).to(device))
    modified_array, sorted_keys = steganography.fgsm_qmdct_modify(cover_array[start:end], grads)
    result['original'] += list(original_probs)

    for key in sorted_keys:
      modified_probs = list(model.get_probabilities(utils.transform(modified_array[key]), [0] * batch_size))
      if key not in result.keys():
        result[key] = modified_probs
      else:
        result[key] += modified_probs

  with open('stego_fgsm_qmdct_modify.csv', 'wt') as f:
    f.write('original_prob,' + ','.join(sorted_keys) + '\n')
    for i in range(len_files):
      s = str(result['original'][i]) + ',' + ','.join(
        [str(result[key][i]) for key in sorted_keys]) + '\n'
      f.write(s)

  print('saved results!')


if __name__ == '__main__':
#  test_gradient_value_guided_qmdct_modify(model='rhfcn', model_path='model_rhfcn.pth', save_path='rhfcn_most.csv', gradient_prefer_type='most')
#  test_gradient_value_guided_qmdct_modify(model='wasdn', model_path='model_wasdn.pth', save_path='wasdn_most.csv', gradient_prefer_type='most')
#  test_gradient_value_guided_qmdct_modify(model='rhfcn', model_path='model_rhfcn.pth', save_path='rhfcn_least.csv', gradient_prefer_type='least')
#  test_gradient_value_guided_qmdct_modify(model='wasdn', model_path='model_wasdn.pth', save_path='wasdn_least.csv', gradient_prefer_type='least')
#  test_fgsm_qmdct_modify()
#  test_two_models()
  test_gradient_value_guided_qmdct_modify('wasdn', 'model_wasdn.pth', neglect_sign=True, save_path='wasdn_neglect.csv')
  test_gradient_value_guided_qmdct_modify('wasdn', 'model_wasdn.pth', neglect_sign=False, save_path='wasdn_not_neglect.csv')
