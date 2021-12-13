#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn
import torch.nn.functional as F

def xavier_init(layer):
  if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
    nn.init.xavier_normal_(layer.weight)

def batch_normalizaiton(num_features, dim=2):
  if dim==2:
    return nn.BatchNorm2d(num_features, momentum=.1, affine=True, eps=1e-5).train()
  else:
    return nn.BatchNorm1d(num_features, momentum=.1, affine=True, eps=1e-5).train()

class WASDN(nn.Module):
  def __init__(self, init=False):
    super(WASDN, self).__init__()
    self.tanh = nn.Tanh()

    # group 1
    self.conv1_1 = nn.Conv2d(9, 16, 3, 1, padding='same')
    self.conv1_2 = nn.Conv2d(16, 32, 1, 1, padding='same')
    self.pool1_3 = nn.MaxPool2d(2, 2)

    # group 2
    self.conv2_1 = nn.Conv2d(32, 32, 3, 1, padding='same')
    self.conv2_2 = nn.Conv2d(32, 64, 1, 1, padding='same')
    self.pool2_3 = nn.MaxPool2d(2, 2)

    # group3
    self.conv3_1 = nn.Conv2d(64, 64, 3, 1, padding='same')
    self.conv3_2 = nn.Conv2d(64, 128, 1, 1, padding='same')
    self.pool3_3 = nn.MaxPool2d(2, 2)

    # group 4
    self.conv4_1 = nn.Conv2d(128, 128, 3, 1, padding='same')
    self.conv4_2 = nn.Conv2d(128, 256, 1, 1, padding='same')
    self.bn4_3 = batch_normalizaiton(256)
    self.pool4_4 = nn.MaxPool2d(2, 2)

    # group 5
    self.conv5_1 = nn.Conv2d(256, 256, 3, 1, padding='same')
    self.conv5_2 = nn.Conv2d(256, 512, 1, 1, padding='same')
    self.bn5_3 = batch_normalizaiton(512)
    self.pool5_4 = nn.MaxPool2d(2, 2)

    # group 6
    self.conv6_1 = nn.Conv2d(512, 512, 3, 1, padding='same')
    self.conv6_2 = nn.Conv2d(512, 1024, 1, 1, padding='same')
    self.bn6_3 = batch_normalizaiton(1024)
    self.pool6_4 = nn.MaxPool2d(2, 2)

    # group 7 -- fully connected layers
    self.fc7_1 = nn.Linear(21504, 4096)
    self.bn7_2 = batch_normalizaiton(4096, dim=1)
    self.fc7_3 = nn.Linear(4096, 512)
    self.bn7_4 = batch_normalizaiton(512, dim=1)
    self.fc7_5 = nn.Linear(512, 2)

    # group all together
    self.group1 = nn.Sequential(self.conv1_1, self.tanh, self.conv1_2, self.tanh, self.pool1_3)
    self.group2 = nn.Sequential(self.conv2_1, self.tanh, self.conv2_2, self.tanh, self.pool2_3)
    self.group3 = nn.Sequential(self.conv3_1, self.tanh, self.conv3_2, self.tanh, self.pool3_3)
    self.group4 = nn.Sequential(self.conv4_1, self.tanh, self.conv4_2, self.bn4_3, self.tanh, self.pool4_4)
    self.group5 = nn.Sequential(self.conv5_1, self.tanh, self.conv5_2, self.bn5_3, self.tanh, self.pool5_4)
    self.group6 = nn.Sequential(self.conv6_1, self.tanh, self.conv6_2, self.bn6_3, self.tanh, self.pool6_4)
    self.group7 = nn.Sequential(self.fc7_1, self.bn7_2, self.tanh, self.fc7_3, self.bn7_4, self.tanh, self.fc7_5)

    if init:
      self.apply(xavier_init)

  def forward(self, input_data):
    temp = self.group1(input_data)
    for idx, layers in enumerate([self.group2, self.group3, self.group4, self.group5, self.group6]):
      temp = layers(temp)

    temp = temp.reshape(temp.shape[0], -1)
    return self.group7(temp)

  def get_probabilities(self, input_data, labels):
    self.eval()
    with T.no_grad():
      probs = F.softmax(self.forward(input_data), dim=1).cpu()
      index = T.LongTensor(labels).unsqueeze(1)
      probs = T.gather(probs, 1, index).squeeze().detach().numpy()
      return probs

class RHFCN(nn.Module):
  def __init__(self, init=False):
    super(RHFCN, self).__init__()
    self.tanh = nn.Tanh()

    # group 1
    self.conv1_1 = nn.Conv2d(9, 16, 3, 1, padding='same')
#    self.conv1_2 = nn.Conv2d(16, 32, 1, 1, padding='same')
    self.conv1_2 = nn.Conv2d(16, 32, 3, 1, padding='same')
    self.bn1_3 = batch_normalizaiton(32)
    self.pool1_4 = nn.MaxPool2d(2, 2)

    # group 2
    self.conv2_1 = nn.Conv2d(32, 32, 3, 1, padding='same')
#    self.conv2_2 = nn.Conv2d(32, 64, 1, 1, padding='same')
    self.conv2_2 = nn.Conv2d(32, 64, 3, 1, padding='same')
    self.bn2_3 = batch_normalizaiton(64)
    self.pool2_4 = nn.MaxPool2d(2, 2)

    # group3
    self.conv3_1 = nn.Conv2d(64, 64, 3, 1, padding='same')
#    self.conv3_2 = nn.Conv2d(64, 128, 1, 1, padding='same')
    self.conv3_2 = nn.Conv2d(64, 128, 3, 1, padding='same')
    self.bn3_3 = batch_normalizaiton(128)
    self.pool3_4 = nn.MaxPool2d(2, 2)

    # group 4
    self.conv4_1 = nn.Conv2d(128, 128, 3, 1, padding='same')
    self.conv4_2 = nn.Conv2d(128, 256, 1, 1, padding='same')
    self.bn4_3 = batch_normalizaiton(256)
    self.pool4_4 = nn.MaxPool2d(2, 2)

    # group 5
    self.conv5_1 = nn.Conv2d(256, 256, 3, 1, padding='same')
    self.conv5_2 = nn.Conv2d(256, 512, 1, 1, padding='same')
    self.bn5_3 = batch_normalizaiton(512)
    self.pool5_4 = nn.MaxPool2d(2, 2)

    # group 6
    self.conv6_1 = nn.Conv2d(512, 4096, [6, 14], 1, padding='valid')
    self.bn6_2 = batch_normalizaiton(4096)

    # group7
    self.conv7_1 = nn.Conv2d(4096, 512, 1, 1, padding='valid')
    self.bn7_2 = batch_normalizaiton(512)

    # group 8
    self.conv8_1 = nn.Conv2d(512, 2, 1, 1, padding='valid')
    self.bn8_2 = batch_normalizaiton(2)

    # group all together
    self.group1 = nn.Sequential(self.conv1_1, self.tanh, self.conv1_2, self.bn1_3, self.tanh, self.pool1_4)
    self.group2 = nn.Sequential(self.conv2_1, self.tanh, self.conv2_2, self.bn2_3, self.tanh, self.pool2_4)
    self.group3 = nn.Sequential(self.conv3_1, self.tanh, self.conv3_2, self.bn3_3, self.tanh, self.pool3_4)
    self.group4 = nn.Sequential(self.conv4_1, self.tanh, self.conv4_2, self.bn4_3, self.tanh, self.pool4_4)
    self.group5 = nn.Sequential(self.conv5_1, self.tanh, self.conv5_2, self.bn5_3, self.tanh, self.pool5_4)
    self.group6 = nn.Sequential(self.conv6_1, self.bn6_2, self.tanh)
    self.group7 = nn.Sequential(self.conv7_1, self.bn7_2, self.tanh)
    self.group8 = nn.Sequential(self.conv8_1, self.bn8_2, self.tanh)

    # global max pooling and logits
    self.max_pool = nn.MaxPool2d(1, 1)

    if init:
      self.apply(xavier_init)

  def forward(self, input_data):
    temp = self.group1(input_data)
    for idx, layers in enumerate([self.group2, self.group3, self.group4, self.group5, self.group6, self.group7, self.group8]):
      temp = layers(temp)

    temp = self.max_pool(temp)
    return temp.reshape(temp.size()[0], 2)

  def get_probabilities(self, input_data, labels):
    self.eval()
    with T.no_grad():
      probs = F.softmax(self.forward(input_data), dim=1).cpu()
      index = T.LongTensor(labels).unsqueeze(1)
      probs = T.gather(probs, 1, index).squeeze().detach().numpy()
      return probs
