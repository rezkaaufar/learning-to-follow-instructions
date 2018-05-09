import numpy as np
from numpy import random
import torch
from torch.autograd import Variable

class Dataset(object):
  def __init__(self, inps, instr, targets, inps_v, instr_v, targets_v, inps_t, instr_t, targets_t):
    """
    Builds dataset with train, test, and validation input and targets.
    It is built to generate random batch for training
    Args:
      inps: List of Initial Configuration and Utterance
      labels: List of Resulting Configuration
    """
    self.inps = inps
    self.instr = instr
    self.targets = targets
    self.inps_t = inps_t
    self.instr_t = instr_t
    self.targets_t = targets_t
    self.inps_v = inps_v
    self.instr_v = instr_v
    self.targets_v = targets_v
    ## get total character ##
    all_characters = set()
    for el in inps:
        els = el.split(" ")
        for e in els:
            all_characters.add(e)
    all_words = set()
    for el in instr:
      els = el.split(" ")
      for e in els:
        all_words.add(e)

    self.all_characters = sorted(list(all_characters))
    self.n_letters = len(all_characters)
    self.all_words = sorted(list(all_words))
    self.n_words = len(all_words)
    # length of the blocks configuration and utterance
    self.len_example = len(inps[0].split(" "))
    self.len_targets = len(targets[0].split(" "))
    self.len_instr = len(instr[0].split(" "))

  def randomize_data(self):
    ## randomize training data ##
    even_ind = np.arange(0, len(self.inps), 2)
    odd_ind = np.arange(1, len(self.inps), 2)
    assert len(even_ind) == len(odd_ind)
    random.shuffle(even_ind)
    random.shuffle(odd_ind)
    shuffled_index = np.array([val for pair in zip(even_ind, odd_ind) for val in pair])
    self.inps = np.array(self.inps)[shuffled_index].tolist()
    self.targets = np.array(self.targets)[shuffled_index].tolist()
    self.instr = np.array(self.instr)[shuffled_index].tolist()
    ## randomize testing data ##
    even_ind = np.arange(0, len(self.inps_t), 2)
    odd_ind = np.arange(1, len(self.inps_t), 2)
    assert len(even_ind) == len(odd_ind)
    random.shuffle(even_ind)
    random.shuffle(odd_ind)
    shuffled_index = np.array([val for pair in zip(even_ind, odd_ind) for val in pair])
    self.inps_t = np.array(self.inps_t)[shuffled_index].tolist()
    self.targets_t = np.array(self.targets_t)[shuffled_index].tolist()
    self.instr_t = np.array(self.instr_t)[shuffled_index].tolist()
    ## randomize validation data ##
    even_ind = np.arange(0, len(self.inps_v), 2)
    odd_ind = np.arange(1, len(self.inps_v), 2)
    assert len(even_ind) == len(odd_ind)
    random.shuffle(even_ind)
    random.shuffle(odd_ind)
    shuffled_index = np.array([val for pair in zip(even_ind, odd_ind) for val in pair])
    self.inps_v = np.array(self.inps_v)[shuffled_index].tolist()
    self.targets_v = np.array(self.targets_v)[shuffled_index].tolist()
    self.instr_v = np.array(self.instr_v)[shuffled_index].tolist()

  # turn string into list of longs
  def char_tensor(self, string):
    string = string.split(" ")
    tensor = torch.zeros(self.n_letters).long()
    char_index = self.all_characters.index(string)
    tensor[char_index] = 1
    return tensor

  def categories_tensor(self, string):
    string = string.split(" ")
    tensor = torch.zeros(len(string)).long()
    for li, letter in enumerate(string):
      letter_index = self.all_characters.index(letter)
      tensor[li] = letter_index
    return tensor

  def seq_tensor(self, string):
    string = string.split(" ")
    tensor = torch.zeros(len(string)).long()
    for li, letter in enumerate(string):
      letter_index = self.all_characters.index(letter)
      tensor[li] = letter_index
    return tensor

  def instr_tensor(self, string):
    string = string.split(" ")
    tensor = torch.zeros(len(string)).long()
    for li, letter in enumerate(string):
      letter_index = self.all_words.index(letter)
      tensor[li] = letter_index
    return tensor

  def generate_batch(self, start_index, batch_size, inps, instrs, labels):
    inp_tensor = torch.zeros(batch_size, self.len_example).long()
    lab_tensor = torch.zeros(batch_size, self.len_targets).long()
    ins_tensor = torch.zeros(batch_size, self.len_instr).long()
    for i in range(batch_size):
      inp = self.seq_tensor(inps[start_index + i])
      lab = self.categories_tensor(labels[start_index + i])
      ins = self.instr_tensor(instrs[start_index + i])
      inp_tensor[i, :] = inp
      lab_tensor[i, :] = lab
      ins_tensor[i, :] = ins
    # uncomment to do this with CPU
    # return Variable(inp_tensor), Variable(lab_tensor)
    return Variable(inp_tensor).cuda(), Variable(ins_tensor).cuda(), Variable(lab_tensor).cuda()

## testing ##
def read_data(path):
  inputs = []
  targets = []
  instrs = []
  with open(path, "r") as f:
    for line in f:
      ar = line.split("\t")
      inp = ar[0]
      ins = ar[1]
      lab = ar[2].replace("\n", "")
      inputs.append(inp)
      targets.append(lab)
      instrs.append(ins)
  return inputs, instrs, targets