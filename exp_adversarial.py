import torch as T
import numpy as np
import tqdm

import steganography
import utils
import config
import itertools

import gradient

def get_single_point_modify_result(torch_model, data, device):
  """
  test for all points' modify

  :param torch_model: RHFCN or WASDN model
  :param data: ndarray, [height, width]
  :param device: str
  """
  data = np.array(data)
  data[200:] = 0
  data[:, 450] = 0
  batch_size = 100
  x_indices, y_indices = np.where(abs(data)>=3)
  results = []
  for i in tqdm.trange(0, len(x_indices), batch_size):
    temp = np.zeros((batch_size, 200, 576, 1))
    temp[:, :, :, 0] = data
    batch_size = batch_size if len(x_indices) >= i+batch_size else len(x_indices) - i
    for batch_i in range(batch_size):
      x, y = x_indices[i + batch_i], y_indices[i + batch_i]
      temp[batch_i, x, y, 0] = -temp[batch_i, x, y, 0]

    results += list(torch_model.get_probabilities(utils.transform(temp, device), [0]*temp.shape[0]))

  return np.array(results)

def get_accumulative_modify_result(torch_model, data, device, baseline, increment_floor):
  """
  test for all points' modify

  :param torch_model: RHFCN or WASDN model
  :param data: ndarray, [height, width]
  :param device: str
  :param baseline: float, original detection probability for cover
  :param increment_floor: float, minimum incremental detection probability to performing modification
  """
  data = np.array(data)
  data[200:] = 0
  data[:, 450] = 0
  batch_size = 10
  x_indices, y_indices = np.where(abs(data)>=3)
  cnt = 0
  for i in tqdm.trange(0, len(x_indices), batch_size):
    temp = np.zeros((batch_size, 200, 576, 1))
    temp[:, :, :, 0] = data
    batch_size = batch_size if len(x_indices) >= i+batch_size else len(x_indices) - i
    for batch_i in range(batch_size):
      x, y = x_indices[i + batch_i], y_indices[i + batch_i]
      temp[batch_i, x, y, 0] += 1

    differences = torch_model.get_probabilities(utils.transform(temp, device), [0]*temp.shape[0]) - baseline
    best_diff, best_batch = differences.max(), differences.argmax()
    if best_diff >= increment_floor:
      baseline += best_diff
      data = temp[best_batch, :, :, 0]
      cnt += 1

  return cnt, baseline

def get_fgsm_modify_result(torch_model, data, gradient, max_modifications, device, amplitude=1):
  """
  test for traditional fgsm adversarial examples

  :param torch_model: RHFCN or WASDN model
  :param data: ndarray, [batch, height, width]
  :param gradient: ndarray, [batch, height, width] which should be all the same with data
  :param max_modifications: list, number of points will be modified(-1 means all)
  :param device, str
  """
  # let all the gradients in small-value zone to be .0
  index = abs(data)<=2 # notice that this logical expression is different with steganography.py
  gradient[index] = .0
  batch, height, width = data.shape
  flattened_gradient = gradient.reshape(batch, -1)
  flattened_index = np.flip(np.argsort(abs(flattened_gradient), axis=1), axis=1)

  results = {}
  for mm in max_modifications:
    temp_data = np.array(data.reshape(batch, -1))
    for i_batch in range(batch):
      indices = flattened_index[i_batch, :mm]
      temp_data[i_batch, indices] -= np.sign(flattened_gradient[i_batch, indices]) * amplitude
    # transform data and pass through the model to get result
    temp_data = temp_data.reshape(batch, height, width)
    temp_data[index] = data[index]
    steganography.assert_modify_correct(data, temp_data, 'big')
    temp_data = np.expand_dims(temp_data, -1) # add channel dimension
    temp_tensor = utils.transform(temp_data, device)
    results[mm] = list(torch_model.get_probabilities(temp_tensor, [0]*batch))

  return results

def _exp_preparation(model, model_path, batch_size, filename=None):
  device = utils.auto_select_device()
  model = utils.load_model(model, model_path, device)
  _, stego_path = config.get_paths('test', 'ags', '2')
  stego_files = utils.get_files_list(stego_path)
  fid = open(filename, 'wt') if filename else None
  def generator():
    for idx in range(0, len(stego_files), batch_size):
      print(f'************************{idx}->{idx+batch_size}loading*********************')
      array_stego = utils.text_read_batch(stego_files[idx:idx+batch_size], progress=True)
      tensor_stego = utils.transform(array_stego, device)
      probabilities = model.get_probabilities(tensor_stego, [0]*array_stego.shape[0])
      del tensor_stego
      yield array_stego, probabilities
  return model, device, fid, generator()

def exp_single_point(model, model_path, batch_size=3):
  model, device, fid, generator = _exp_preparation(model, model_path, batch_size)
  file_index = 0
  for array_stego, probabilities in generator:
    for i in range(batch_size):
      file_index += 1
      print(f'-------------------{file_index}th iteration--------------------------------')
      modified_probabilities = get_single_point_modify_result(model, array_stego[i, :, :, 0], device)
      differences = modified_probabilities - probabilities[i]
      differences.sort()
      print(f'prob: {probabilities[i]} total: {len(differences)}')
      print(f'min: {differences[:10]}\nmax: {differences[-10:]}')
      for k, g in itertools.groupby(differences, lambda x: int(x*100)):
        print(f'impact {k/100} #:{len(list(g))}')

def exp_accumulative_modify(model, model_path, batch_size=50, filename='accumulative_modify.txt'):
  model, device, fid, generator = _exp_preparation(model, model_path, batch_size, filename)
  fid.write('prob,modifications,modified_prob,diff\n')
  file_index = 0
  for array_stego, probabilities in generator:
    for i in range(batch_size):
      file_index += 1
      print(f'-------------------{file_index}th iteration--------------------------------')
      counter, modified_prob = get_accumulative_modify_result(model, array_stego[i, :, :, 0], device, probabilities[i], 0.0001)
      difference = modified_prob - probabilities[i]
      print(f'prob: {probabilities[i]} modifications: {counter} modified prob: {modified_prob} diff: {difference}')
      fid.write(f'{probabilities[i]},{counter},{modified_prob},{difference}\n')
      fid.flush()
  fid.close()

def exp_fgsm_modify(model, model_path, batch_size=50, max_modifications=[100, 500, 1000, 5000, 10000, 20000, 30000, -1], amplitude=100, filename='fgsm_modify.csv'):
  model, device, fid, generator = _exp_preparation(model, model_path, batch_size, filename)
  fid.write('prob,'+','.join([str(mm) for mm in max_modifications])+'\n')
  batch_index = 0
  for array_stego, probabilities in generator:
    batch_index += 1
    print(f'------------------------{batch_index}th batch iteration------------------')
    _, grads = gradient.data_gradient(model, array_stego, T.LongTensor([0]*batch_size).to(device), device=device)
    dict_results = get_fgsm_modify_result(model, array_stego[:, :, :, 0], grads[:, :, :, 0], max_modifications, device, amplitude=amplitude)
    dict_differences = {}
    for key in dict_results.keys():
      dr = np.array(dict_results[key]) - probabilities
#      dr[abs(dr)<=0.01] = 0
      dict_differences[key] = dr
    for i in range(batch_size):
      fid.write(f'{probabilities[i]},'+','.join([str(dict_differences[mm][i]) for mm in max_modifications])+'\n')
    fid.flush()
  fid.close()

if __name__ == '__main__':
#  exp_single_point('rhfcn', 'model_rhfcn_local.pth')
#  exp_accumulative_modify('rhfcn', 'multi_stego_rhfcn.pth')
  for amplitude in [0.1, 0.2, 0.5, 1, 2, 10, 20, 50, 100]:
    exp_fgsm_modify('wasdn', 'multi_stego_wasdn.pth', amplitude=amplitude, filename=f'fgsm_modify_multi_stego_wasdn_amplitude_{amplitude}.csv')