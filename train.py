#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dataset
import utils
from network import RHFCN, WASDN
import torch as T
import torch.utils.tensorboard as tensorboard
import torch.nn as nn
import torch.optim as O
import numpy as np
import tqdm
import config
from dataset import PairDataset

class NestedBreakException(Exception):
  pass

def evaluation(torch_model, dataset, device, update_progress_hook=lambda: None):
  correct = []
  for data, labels in dataset:
    data = utils.transform(data, device)
    with T.no_grad():
      torch_model.eval()
      _, predictions = T.max(torch_model(data).cpu(), axis=1)
    torch_model.train()
    correct += list((predictions == T.LongTensor(labels)).numpy().astype(int))
    update_progress_hook()

  return correct, sum(correct) / len(correct)

def cross_validation_train(model, cover_files, stego_files, epochs=4, batch_size=10, save_path=None, smoothed_weight=0.3, exit_threhold=0.8):
  """
  should not be used any longer! because the PairDataset class is re-designed.
  """
  writer = tensorboard.SummaryWriter(f'runs/train_{utils.get_time()}')
  device = utils.auto_select_device()
  dataset = PairDataset(cover_files, stego_files, folds=10, batch_size=batch_size, to_tensor=ToTensor(device))

  if model == 'wasdn':
    model = WASDN().to(device)
  elif model == 'rhfcn':
    model = RHFCN().to(device)

  optimizer = O.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
  criterion = nn.CrossEntropyLoss().to(device)

  try:
    pbar = tqdm.tqdm(total=epochs * dataset.iterations())
    smoothed_accuracy = None
    for epoch in range(epochs):
      for idx, (tag, data, labels) in enumerate(dataset):
        # training here
        if tag == 'T':
          loss = criterion(model(data), T.LongTensor(labels).to(device))
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          writer.add_scalar('Loss', loss.cpu().detach().item(), epoch * dataset.iterations() + idx)

        elif tag == 'V':
          correct, accuracy = evaluation(model, data, labels, batch_size)
          writer.add_scalar('Accuracy', accuracy, epoch * dataset.iterations() + idx)

          if not smoothed_accuracy:
            smoothed_accuracy = accuracy
          else:
            smoothed_accuracy = smoothed_accuracy * smoothed_weight + (1-smoothed_weight) * accuracy

          if smoothed_accuracy >= exit_threhold:
            raise NestedBreakException()

        else:
          raise ValueError('PairDataset returned tag is not T or V.')

        pbar.update(1)
  except NestedBreakException:
    pass

  utils.save_model(model, save_path)
  print(f'model saved to {save_path}')
  writer.flush()
  writer.close()

def multi_stego_train(model, cover_files, stego_files_list, train_size=0.9, epochs=1, batch_size=20, save_path=None, device=None):
  """
  use multiple stego files train model

  :param model: str, 'rhfcn' or 'wasdn'
  :param cover_files: list[str]
  :param stego_files_list: list[list[str], ]
  :param train_size: float, 0-1.0
  :param epochs: int
  :param batch_size: int
  :param save_path: str
  """
  writer = tensorboard.SummaryWriter(f'runs/train_{utils.get_time()}')
  device = device if device else utils.auto_select_device()
  if model == 'rhfcn':
    model = RHFCN().to(device)
  elif model == 'wasdn':
    model = WASDN().to(device)

  files_len = len(cover_files)
  train_len = int(files_len * train_size)
  train_dataset = dataset.MultiStegoPairDataset(cover_files[:train_len], [s[:train_len] for s in stego_files_list], batch_size=batch_size)
  valid_dataset = dataset.MultiStegoPairDataset(cover_files[train_len:], [s[train_len:] for s in stego_files_list], batch_size=batch_size)
  optimizer = O.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
  criterion = nn.CrossEntropyLoss().to(device)
  try:
    pbar = tqdm.tqdm(total=epochs * (train_dataset.iterations()+valid_dataset.iterations()))
    update_progress_hook = lambda: pbar.update(1)
    for epoch in range(epochs):
      # train procedures for current epoch
      train_dataset = iter(train_dataset)
      for train_data, train_labels in train_dataset:
        train_data = utils.transform(train_data, device)
        train_labels = T.LongTensor(train_labels).to(device)
        loss = criterion(model(train_data), train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss', loss.cpu().detach().item())
        update_progress_hook()

      # validation procedures for current epoch
      valid_dataset = iter(valid_dataset)
      _, accuracy = evaluation(model, valid_dataset, device, update_progress_hook)
      writer.add_scalar('Acc', accuracy)

  except NestedBreakException:
    pass

  utils.save_model(model, save_path)

  writer.flush()
  writer.close()

def test(model, model_path, cover_files, stego_files_list, detailed_result=None, device=None):
  device = device if device else utils.auto_select_device()
  model = utils.load_model(model, model_path, device)
  test_dataset = dataset.MultiStegoPairDataset(cover_files, stego_files_list, batch_size=10)
  pbar = tqdm.tqdm(total=test_dataset.iterations())
  correct, accuracy = evaluation(model, test_dataset, device, lambda: pbar.update(1))
  if detailed_result:
    dr_file = open(detailed_result, 'wt')
    results_per_file = len(stego_files_list) + 1
    for i_file in range(len(test_dataset)):
      temp = ','.join([str(x) for x in correct[i_file*results_per_file:(i_file+1)*results_per_file]])
      dr_file.write(f'{i_file+1},{temp}\n')
    dr_file.write(f'total,{accuracy}\n')

  return correct, accuracy

# experiments
def _exp_train_multi_stego_model():
  cover_path, stego_paths = config.get_multi_stego_paths('train', ['jed', 'ags-small', 'ags-small', 'ags-big', 'ags-big'], [2, 8000, 20000, 8000, 20000])
  cover_files = utils.get_files_list(cover_path)
  stego_files_list = [utils.get_files_list(path) for path in stego_paths]
  multi_stego_train('rhfcn', cover_files, stego_files_list,
                    train_size=0.98,
                    epochs=3,
                    batch_size=10,
                    save_path='multi_stego_rhfcn.pth')

def _exp_test_multi_stego_model():
  cover_path, stego_paths = config.get_multi_stego_paths('train', ['jed', 'ags-small', 'ags-small', 'ags-big', 'ags-big'], [2, 8000, 20000, 8000, 20000])
  cover_files = utils.get_files_list(cover_path)
  stego_files_list = [utils.get_files_list(path) for path in stego_paths]
  test('rhfcn', 'multi_stego_rhfcn.pth', cover_files, stego_files_list, 'train_dataset_result.csv')
  cover_path, stego_paths = config.get_multi_stego_paths('test', ['jed']*5+['ags']*5, [2,4,6,8,10]*2)
  cover_files = utils.get_files_list(cover_path)
  stego_files_list = [utils.get_files_list(path) for path in stego_paths]
  test('rhfcn', 'multi_stego_rhfcn.pth', cover_files, stego_files_list, 'test_dataset_result.csv')

if __name__ == '__main__':
  _exp_test_multi_stego_model()
