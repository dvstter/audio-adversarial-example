#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T

class Filters:
  def __init__(self, height, width):
    # usage: matmal(data, _1)
    self._1 = T.eye(width)
    for y in range(width-1):
      self._1[:,y] -= self._1[:,y+1]

    # usage: matmal(_2, data)
    self._2 = T.eye(height)
    for x in range(height-1):
      self._2[x,:] -= self._2[x+1,:]

    # usage: matmal(data, _1)
    self._3 = T.eye(width)
    for y in range(width-2):
      self._3[:,y] -= 2*self._3[:,y+1] - self._3[:,y+2]

    # usage: matmal(_2, data)
    self._4 = T.eye(height)
    for x in range(height-2):
      self._4[x,:] -= 2*self._4[x+1,:] - self._4[x+2,:]

  def transform(self, input_data):
      batch, _, height, width = input_data.shape
      data = T.zeros([batch, 9, height, width])
      data[:, 0] = input_data[:, 0]
      data[:, 1] = T.matmul(input_data[:, 0], self._1)
      data[:, 2] = T.matmul(self._2, input_data[:, 0])
      data[:, 3] = T.matmul(input_data[:, 0].abs(), self._1)
      data[:, 4] = T.matmul(self._2, input_data[:, 0].abs())
      data[:, 5] = T.matmul(input_data[:, 0], self._3)
      data[:, 6] = T.matmul(self._4, input_data[:, 0])
      data[:, 7] = T.matmul(input_data[:, 0].abs(), self._3)
      data[:, 8] = T.matmul(self._4, input_data[:, 0].abs())
      return data

  def diff_layer(input_data, is_diff, is_diff_abs, is_abs_diff, order, direction, name, padding="SAME"):
    """
    the layer which is used for difference
    :param input_data: the input data tensor [batch_size, height, width, channels]
    :param is_diff: whether make difference or not
    :param is_diff_abs: whether make difference and abs or not
    :param is_abs_diff: whether make abs and difference or not
    :param order: the order of difference
    :param direction: the direction of difference, "inter"(between row) or "intra"(between col)
    :param name: the name of the layer
    :param padding: the method of padding, default is "SAME"

    :return:
        feature_map: 4-D tensor [number, height, width, channel]
    """

    print("name: %s, is_diff: %r, is_diff_abs: %r, is_abs_diff: %r, order: %d, direction: %s"
          % (name, is_diff, is_diff_abs, is_abs_diff, order, direction))

    if order == 0:
      return input_data
    else:
      if order == 1 and direction == "inter":
        filter_diff = tf.constant(value=[1, -1],
                                  dtype=tf.float32,
                                  shape=[2, 1, 1, 1],
                                  name="diff_inter_1")
      elif order == 1 and direction == "intra":
        filter_diff = tf.constant(value=[1, -1],
                                  dtype=tf.float32,
                                  shape=[1, 2, 1, 1],
                                  name="diff_intra_1")
      elif order == 2 and direction == "inter":
        filter_diff = tf.constant(value=[1, -2, 1],
                                  dtype=tf.float32,
                                  shape=[3, 1, 1, 1],
                                  name="diff_inter_2")
      elif order == 2 and direction == "intra":
        filter_diff = tf.constant(value=[1, -2, 1],
                                  dtype=tf.float32,
                                  shape=[1, 3, 1, 1],
                                  name="diff_intra_2")
      else:
        filter_diff = tf.constant(value=[1],
                                  dtype=tf.float32,
                                  shape=[1, 1, 1, 1],
                                  name="None")

      if is_diff is True:
        output = tf.nn.conv2d(input=input_data,
                              filter=filter_diff,
                              strides=[1, 1, 1, 1],
                              padding=padding)

        return output

      elif is_diff_abs is True:
        output = tf.nn.conv2d(input=input_data,
                              filter=filter_diff,
                              strides=[1, 1, 1, 1],
                              padding=padding)
        output = tf.abs(output)

        return output

      elif is_abs_diff is True:
        input_data = tf.abs(input_data)
        output = tf.nn.conv2d(input=input_data,
                              filter=filter_diff,
                              strides=[1, 1, 1, 1],
                              padding=padding)

        return output

      else:
        return input_data


def rich_hpf_layer(input_data, name):
  """
  multiple HPF processing
  diff_layer(input_data, is_diff, is_diff_abs, is_abs_diff, order, direction, name, padding="SAME")

  :param input_data: the input data tensor [batch_size, height, width, channels]
  :param name: the name of the layer
  :return:
      feature_map: 4-D tensor [number, height, width, channel]
  """
  dif_inter_1 = diff_layer(input_data, True, False, False, 1, "inter", "dif_inter_1", padding="SAME")
  dif_inter_2 = diff_layer(input_data, True, False, False, 2, "inter", "dif_inter_2", padding="SAME")
  dif_intra_1 = diff_layer(input_data, True, False, False, 1, "intra", "dif_intra_1", padding="SAME")
  dif_intra_2 = diff_layer(input_data, True, False, False, 2, "intra", "dif_intra_2", padding="SAME")

  dif_abs_inter_1 = diff_layer(input_data, False, False, True, 1, "inter", "abs_dif_inter_1", padding="SAME")
  dif_abs_inter_2 = diff_layer(input_data, False, False, True, 2, "inter", "abs_dif_inter_2", padding="SAME")
  dif_abs_intra_1 = diff_layer(input_data, False, False, True, 1, "intra", "abs_dif_intra_1", padding="SAME")
  dif_abs_intra_2 = diff_layer(input_data, False, False, True, 2, "intra", "abs_dif_intra_2", padding="SAME")

  output = tf.concat(
    [dif_inter_1, dif_inter_2, dif_intra_1, dif_intra_2, dif_abs_inter_1, dif_abs_inter_2, dif_abs_intra_1,
     dif_abs_intra_2], 3, name=name)

  return output
