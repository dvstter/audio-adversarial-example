#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import utils
from network import RHFCN, WASDN
import torch as T
import torch.utils.tensorboard as tensorboard
import torch.nn as nn
import torch.optim as O
import tqdm
import config
from dataset import PairDataset, ToTensor

class NestedBreakException(Exception):
  pass

def evaluation(model, data, labels, batch_size=500, progress=False):
  """
  Evaluate the model

  :param model: nn.Module
  :param data: tensor
  :param labels: tensor, list or ndarray
  :param batch_size: int, to avoid gpu overload
  :param progress: bool, display tqdm progress bar or not

  :return:
    correct, accuracy
  """

  if not T.is_tensor(labels):
    labels = T.LongTensor(labels)

  num_data = data.shape[0]
  batches = num_data // batch_size if batch_size <= num_data else 1
  correct = 0
  _range = tqdm.trange(batches) if progress else range(batches)
  for i in _range:
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_data = data[start_idx:end_idx]
    batch_labels = labels[start_idx:end_idx]
    with T.no_grad():
      model.eval()
      _, predictions = T.max(model(batch_data).cpu(), axis=1)
    model.train()
    correct += (predictions == batch_labels).sum().item()

  return correct, correct / num_data

def cross_validation_train(model, cover_files, stego_files, epochs=4, batch_size=10, save_path=None, smoothed_weight=0.3, exit_threhold=0.8):
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

def train(model, cover_files, stego_files, train_size=0.9, epochs=1, batch_size=10, save_path=None):
  """
  This method has been deprecated.
  """
  writer = tensorboard.SummaryWriter(f'runs/train_{utils.get_time()}')
  device = utils.auto_select_device()

  cover_len = len(cover_files)
  train_amount = int(cover_len * train_size)

  train_cover_files, train_stego_files, valid_cover_files, valid_stego_files = cover_files[:train_amount], stego_files[:train_amount], cover_files[train_amount:], stego_files[train_amount:]
  writer.add_text(f'Log', f'Files\' length:\n\ttrain-cover {len(train_cover_files)} train-stego {len(train_stego_files)}\n\tvalid-cover {len(valid_cover_files)} valid-stego {len(valid_stego_files)}', 0)

  train_covers = utils.transform(utils.text_read_batch(train_cover_files, progress=True))
  train_stegos = utils.transform(utils.text_read_batch(train_stego_files, progress=True))
  valid_covers = utils.transform(utils.text_read_batch(valid_cover_files, progress=True))
  valid_stegos = utils.transform(utils.text_read_batch(valid_stego_files, progress=True))
  writer.add_text(f'Log', f'train-covers shape {train_covers} train-stegos shape {train_stegos}\n\tvalid-covers shape {valid_covers} valid-stegos shape {valid_stegos}', 1)

  if model == 'rhfcn':
    model = RHFCN().to(device)
  elif model == 'wasdn':
    model = WASDN().to(device)

  optimizer = O.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
  criterion = nn.CrossEntropyLoss().to(device)
  batches = train_amount // batch_size
  valid_amount = len(cover_files) - train_amount

  valid_data = T.cat([valid_covers, valid_stegos])
  valid_labels = T.LongTensor([0] * valid_amount + [1] * valid_amount)

  try:
    with tqdm.tqdm(total=epochs * batches) as pbar:
      for epoch in range(epochs):
        for i in range(batches):
          if i % 10 == 0:
            correct, accuracy = evaluation(model, valid_data, valid_labels, valid_amount)
            writer.add_scalar('Accuracy', accuracy, epoch * batches + i)

#            if accuracy > 0.92:
#              print('accuracy is higher than .92, exit early.')
#              raise NestedBreakException

          start_idx = i * batch_size
          end_idx = start_idx + batch_size

          train_data = T.cat([train_covers[start_idx:end_idx], train_stegos[start_idx: end_idx]])
          train_labels = T.LongTensor([0]*batch_size+[1]*batch_size).to(device)

          loss = criterion(model(train_data), train_labels)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          writer.add_scalar('Loss', loss.cpu().detach().item(), epoch * batches + i)
          pbar.update(1)
  except NestedBreakException:
    pass

  utils.save_model(model, save_path)

  writer.flush()
  writer.close()

def test(model='rhfcn', model_path='model_rhfcn.pth', birate=320, verbose=True):
  cover_path, stego_path = config.get_paths(birate=birate)
  device = utils.auto_select_device()

  cover_files, stego_files = utils.get_files_list(cover_path), utils.get_files_list(stego_path)
  covers = utils.transform(utils.text_read_batch(cover_files, progress=True))
  stegos = utils.transform(utils.text_read_batch(stego_files, progress=True))

  if verbose:
    print('files loaded.')
    print(f'len cover files: {len(cover_files)} len stego files: {len(stego_files)}')
    print(f'dimension covers: {covers.shape} dimension stegos: {stegos.shape}')
    print(f'prepare loading model {model} from {model_path}')

  model = utils.load_model(model, model_path, device)

  if len(cover_files) > 0:
    correct1, accuracy1 = evaluation(model, covers, [0]*len(cover_files), 500, progress=True)
    print(f'cover : correct {correct1} accuracy {accuracy1}')
  else:
    correct1, accuracy1 = 0, 0

  if len(stego_files) > 0:
    correct2, accuracy2 = evaluation(model, stegos, [1]*len(stego_files), 500, progress=True) if len(stego_files) > 0 else (0, 0)
    print(f'stego : correct {correct2} accuracy {accuracy2}')
  else:
    correct2, accuracy2 = 0, 0

  total_accuracy = (correct1 + correct2) / (len(cover_files)+len(stego_files))

  return correct1 + correct2, total_accuracy

if __name__ == '__main__':
  cover_path, stego_path = config.get_paths(birate=320)
  cover_files, stego_files = utils.get_files_list(cover_path), utils.get_files_list(stego_path)
  train('rhfcn', cover_files[:5000], stego_files[:5000], save_path='model_rhfcn_local.pth')
  test(model='rhfcn', model_path='model_rhfcn_local.pth')
