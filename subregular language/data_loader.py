import numpy as np
from numpy import random
import torch
from torch.autograd import Variable

class Dataset(object):
  def __init__(self, inps, labels, inps_v, labels_v, inps_t, labels_t):
    """
    Builds dataset with train and test input and labels.
    It is built to generate random batch for training
    Args:
      inps: List of grammar (consists of k-factors and target string)
      labels: Labels data
    """
    self._all_characters = ['a', 'b', 'c', 'd', 'e', 'f', '#', '$']
    self._n_letters = 8
    self._categ = ['0', '1']
    self._n_categories = 2
    self._len_example = 40  # length of the grammars and example
    self._inps = inps
    self._labels = labels
    self._inps_t = inps_t
    self._labels_t = labels_t
    self._inps_v = inps_v
    self._labels_v = labels_v

  def randomize_data(self):
    ## randomize training data ##
    even_ind = np.arange(0, len(self._inps), 2)
    odd_ind = np.arange(1, len(self._inps), 2)
    assert len(even_ind) == len(odd_ind)
    random.shuffle(even_ind)
    random.shuffle(odd_ind)
    shuffled_index = np.array([val for pair in zip(even_ind, odd_ind) for val in pair])
    self._inps = np.array(self._inps)[shuffled_index].tolist()
    self._labels = np.array(self._labels)[shuffled_index].tolist()
    ## randomize testing data ##
    even_ind = np.arange(0, len(self._inps_t), 2)
    odd_ind = np.arange(1, len(self._inps_t), 2)
    assert len(even_ind) == len(odd_ind)
    random.shuffle(even_ind)
    random.shuffle(odd_ind)
    shuffled_index = np.array([val for pair in zip(even_ind, odd_ind) for val in pair])
    self._inps_t = np.array(self._inps_t)[shuffled_index].tolist()
    self._labels_t = np.array(self._labels_t)[shuffled_index].tolist()
    ## randomize validation data ##
    even_ind = np.arange(0, len(self._inps_v), 2)
    odd_ind = np.arange(1, len(self._inps_v), 2)
    assert len(even_ind) == len(odd_ind)
    random.shuffle(even_ind)
    random.shuffle(odd_ind)
    shuffled_index = np.array([val for pair in zip(even_ind, odd_ind) for val in pair])
    self._inps_v = np.array(self._inps_v)[shuffled_index].tolist()
    self._labels_v = np.array(self._labels_v)[shuffled_index].tolist()

  def randomize_data_odd(self):
    # even_ind = np.arange(0,len(inps),2)
    odd_ind = np.arange(1, len(self._inps), 2)
    # assert len(even_ind) == len(odd_ind)
    # random.shuffle(even_ind)
    random.shuffle(odd_ind)
    shuffled_index = np.array(odd_ind)
    self._inps = np.array(self._inps)[shuffled_index].tolist()
    self._labels = np.array(self._labels)[shuffled_index].tolist()

    # even_ind = np.arange(0,len(inps),2)
    odd_ind = np.arange(1, len(self._inps_v), 2)
    # assert len(even_ind) == len(odd_ind)
    # random.shuffle(even_ind)
    random.shuffle(odd_ind)
    shuffled_index = np.array(odd_ind)
    self._inps_v = np.array(self._inps_v)[shuffled_index].tolist()
    self._labels_v = np.array(self._labels_v)[shuffled_index].tolist()

    # even_ind = np.arange(0,len(inps),2)
    odd_ind = np.arange(1, len(self._inps_t), 2)
    # assert len(even_ind) == len(odd_ind)
    # random.shuffle(even_ind)
    random.shuffle(odd_ind)
    shuffled_index = np.array(odd_ind)
    self._inps_t = np.array(self._inps_t)[shuffled_index].tolist()
    self._labels_t = np.array(self._labels_t)[shuffled_index].tolist()

  def randomize_data_ev(self):
    even_ind = np.arange(0, len(self._inps), 2)
    # odd_ind = np.arange(1,len(inps),2)
    # assert len(even_ind) == len(odd_ind)
    random.shuffle(even_ind)
    # random.shuffle(odd_ind)
    shuffled_index = np.array(even_ind)
    self._inps = np.array(self._inps)[shuffled_index].tolist()
    self._labels = np.array(self._labels)[shuffled_index].tolist()

    even_ind = np.arange(0, len(self._inps_v), 2)
    # odd_ind = np.arange(1,len(inps),2)
    # assert len(even_ind) == len(odd_ind)
    random.shuffle(even_ind)
    # random.shuffle(odd_ind)
    shuffled_index = np.array(even_ind)
    self._inps_v = np.array(self._inps_v)[shuffled_index].tolist()
    self._labels_v = np.array(self._labels_v)[shuffled_index].tolist()

    even_ind = np.arange(0, len(self._inps_t), 2)
    # odd_ind = np.arange(1,len(inps),2)
    # assert len(even_ind) == len(odd_ind)
    random.shuffle(even_ind)
    # random.shuffle(odd_ind)
    shuffled_index = np.array(even_ind)
    self._inps_t = np.array(self._inps_t)[shuffled_index].tolist()
    self._labels_t = np.array(self._labels_t)[shuffled_index].tolist()

  # turn string into list of longs
  def char_tensor(self, string):
    tensor = torch.zeros(self._n_letters).long()
    char_index = self._all_characters.index(string)
    tensor[char_index] = 1
    return tensor

  def categories_tensor(self, string):
    tensor = torch.zeros(len(string)).long()
    for li, letter in enumerate(string):
        letter_index = self._categ.index(letter)
        tensor[li] = letter_index
    return tensor

  def seq_tensor(self, string):
    tensor = torch.zeros(len(string)).long()
    for li, letter in enumerate(string):
        letter_index = self._all_characters.index(letter)
        tensor[li] = letter_index
    return tensor

  def generate_batch(self, start_index, batch_size, train=0):
    inp_tensor = torch.zeros(batch_size, self._len_example).long()
    lab_tensor = torch.zeros(batch_size, 1).long()
    if train == 0:
      for i in range(batch_size):
        inp = self.seq_tensor(self._inps[start_index + i])
        lab = self.categories_tensor(self._labels[start_index + i])
        inp_tensor[i, :] = inp
        lab_tensor[i, :] = lab
    elif train == 1:
      for i in range(batch_size):
        inp = self.seq_tensor(self._inps_v[start_index + i])
        lab = self.categories_tensor(self._labels_v[start_index + i])
        inp_tensor[i, :] = inp
        lab_tensor[i, :] = lab
    else:
      for i in range(batch_size):
        inp = self.seq_tensor(self._inps_t[start_index + i])
        lab = self.categories_tensor(self._labels_t[start_index + i])
        inp_tensor[i, :] = inp
        lab_tensor[i, :] = lab
    # uncomment to do this with CPU
    # return Variable(inp_tensor), Variable(lab_tensor)
    return Variable(inp_tensor, requires_grad=False).cuda(), Variable(lab_tensor, requires_grad=False).cuda()

  def generate_random_batch(self, batch_size, train=True):
    inp_tensor = torch.zeros(batch_size, self._len_example).long()
    lab_tensor = torch.zeros(batch_size, 1).long()
    if train:
      even_ind = np.arange(0, len(self._inps), 2)
      odd_ind = np.arange(1, len(self._inps), 2)
      sz = batch_size // 2
      a = random.choice(even_ind, sz)
      b = random.choice(odd_ind, sz)
      random_index = np.concatenate([a, b])
      for i, ind in enumerate(random_index):
        inp = self.seq_tensor(self._inps[ind])
        lab = self.categories_tensor(self._labels[ind])
        inp_tensor[i, :] = inp
        lab_tensor[i, :] = lab
    else:
      even_ind = np.arange(0, len(self._inps_t), 2)
      odd_ind = np.arange(1, len(self._inps_t), 2)
      a = random.choice(even_ind, batch_size // 2)
      b = random.choice(odd_ind, batch_size // 2)
      random_index = np.concatenate([a, b])
      for i, ind in enumerate(random_index):
        inp = self.seq_tensor(self._inps_t[ind])
        lab = self.categories_tensor(self._labels_t[ind])
        inp_tensor[i, :] = inp
        lab_tensor[i, :] = lab
    # uncomment to do this with CPU
    # return Variable(inp_tensor), Variable(lab_tensor)
    return Variable(inp_tensor, requires_grad=False).cuda(), Variable(lab_tensor, requires_grad=False).cuda()

  def convert_string(self, texts):
    inp_tensor = torch.zeros(1, self._len_example).long()
    inp = self.seq_tensor(texts)
    inp_tensor[0, :] = inp
    # uncomment to do this with CPU
    # return Variable(inp_tensor), Variable(lab_tensor)
    return Variable(inp_tensor, requires_grad=False).cuda()

## testing ##
def read_data(path):
  inputs = []
  labels = []
  with open(path, "r") as f:
    for line in f:
      ar = line.split("\t")
      g_str = ar[0].replace(" ", "#")
      inp = g_str + "$" + ar[1]
      lab = ar[2].replace("\n", "")
      inputs.append(inp)
      labels.append(lab)
  return inputs, labels

