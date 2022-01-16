#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
import numpy as np
import torch as T
import time
import tqdm
import os

from filters import Filters

def get_freer_gpu():
  os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
  memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
  return np.argmax(memory_available)


def auto_select_device():
  if not T.cuda.is_available():
    return 'cpu'
  else:
    return f'cuda:{get_freer_gpu()}'

def get_time(unix_time_stamp=None):
  """
  unix time stamp -> time in "%Y-%m-%d %H:%M:%S" format
  e.g. 1522048036 -> 2018-03-26 15:07:16
  :param unix_time_stamp: unix time stamp
  :return:
      time_string: time in "%Y-%m-%d %H:%M:%S" format
  """
  time_format = "%Y-%m-%d %H:%M:%S"
  if unix_time_stamp is None:
    value = time.localtime()
  else:
    value = time.localtime(unix_time_stamp)
  time_string = time.strftime(time_format, value)

  return time_string

def get_files_list(file_dir, file_type="txt", start_idx=None, end_idx=None):
  """
  get the files list
  :param file_dir: file directory
  :param file_type: type of files, "*" is to get all files in this folder
  :param start_idx: start index
  :param end_idx: end index
  :return:
      file_list: a list containing full file path
  """
  pattern = "/*." + file_type
  file_list = sorted(glob(file_dir + pattern))
  total_num = len(file_list)
  if type(start_idx) is int and start_idx > total_num:
    start_idx = None
  if type(end_idx) is int and end_idx > total_num:
    end_idx = None
  file_list = file_list[start_idx:end_idx]

  return file_list

def text_read(text_file_path, height=200, width=576, channel=1, separator=","):
  """
  data read from one text file

  :param text_file_path: the file path
  :param height: the height of QMDCT matrix
  :param width: the width of QMDCT matri
  :param channel: the channel of QMDCT matrix
  :param separator: separator of each elements in text file

  :return
      content: QMDCT matrix  ndarray, shape: [height, width, 1]
  """
  content = []
  try:
    with open(text_file_path) as file:
      # read data line by line
      lines = file.readlines()
      for line in lines:
        try:
          numbers = [int(character) for character in line.split(separator)[:-1]]
        except ValueError:
          numbers = [float(character) for character in line.split(separator)[:-1]]
        content.append(numbers)

      content = np.array(content)

      # reshape
      [h, w] = np.shape(content)

      height_new = None if h < height else height
      width_new = None if w < width else width

      if channel == 0:
        content = content[:height_new, :width_new]
      else:
        content = np.reshape(content, [h, w, channel])
        content = content[:height_new, :width_new, :channel]

  except ValueError:
    print("Error read: %s" % text_file_path)

  return content

def text_write(text_file_path, data, separator=','):
  """
  write data into file.

  :param text_file_path: str, file's path
  :param data: ndarry, shape: [height, width, 1]
  :param separator: str

  :return:
    None
  """
  height, width, _ = data.shape
  with open(text_file_path, 'wt') as file:
    for y in range(height):
      for x in range(width):
        file.write(f'{data[y,x,0]},')
      file.write('\n')

def text_read_batch(text_files_list, height=200, width=576, separator=",", progress=False):
  """
  read all txt files into the memory

  :param text_files_list: text files list
  :param height: the height of QMDCT matrix
  :param width: the width of QMDCT matrix
  :param channel: the channel of QMDCT matrix
  :param separator: separator of each elements in text file
  :param progress: bool, display progress bar

  :return:
      data: QMDCT matrixs, ndarry, shape: [files_num, height, width, channel]
  """

  files_num = len(text_files_list)
  data = np.zeros([files_num, height, width, 1], dtype=np.float32)

  _range = tqdm.trange(files_num) if progress else range(files_num)
  for i in _range:
    content = text_read(text_files_list[i], height=height, width=width, channel=1, separator=separator)
    data[i] = content

  return data

def text_write_batch(text_files_list, data, separator=','):
  """
  write all data into text files.

  :param text_files_list: list, shape: [files_num]
  :param data: ndarry, shape: [files_num, height, width, 1]
  :param separator: str

  :return:
    None
  """
  for idx, text_file in enumerate(text_files_list):
    text_write(text_file, data[idx], separator=separator)

FILTERS = Filters(200, 576)
def QMDCTtoHPF(input_data):
  return FILTERS.transform(input_data)

def transform(array, device=None):
  if not device:
    device = auto_select_device()
  if array.ndim != 4:
    raise ValueError('array\'s dimension must be 4 [B*H*W*C]')
  tensor = T.FloatTensor(array)
  tensor = tensor.permute(0,3,1,2)
  tensor = QMDCTtoHPF(tensor)
  tensor = tensor[:, :, :200, :450]
  return tensor.to(device)

def save_model(model, path=None):
  if not path:
    path = f'model_{get_time()}.pth'
  T.save(model.state_dict(), path)
  print(f'Saved model to {path}')

from network import RHFCN, WASDN
def load_model(model, path, device=None):
  if not device:
    device = auto_select_device()

  if model == 'rhfcn':
    model = RHFCN()
  elif model == 'wasdn':
    model = WASDN()
  else:
    raise ValueError('load_modle\' parameter model should only be rhfcn or wasdn.')

  model.load_state_dict(T.load(path))
  model = model.to(device)
  model.eval()
  return model

def convert_text_format(filedir, height, width, separator='\t', replace_files=True, progress=True):
  files = get_files_list(filedir)
  outfiles = [x+'.modif' for x in files]
  array = text_read_batch(files, height=height, width=width, progress=True, separator=separator)
  newarray = np.zeros([500, 200, 576, 1], np.int)
  newarray[:, :, :450, :] = array
  text_write_batch(outfiles, newarray)
  if replace_files:
    for i in tqdm.trange(len(files)):
      os.remove(files[i])
      os.rename(outfiles[i], files[i])
