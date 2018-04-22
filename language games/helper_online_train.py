import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import Decoder
import Encoder
import data_loader
import train
import evaluation
import random
import math
import itertools
import numpy as np
import copy
from tensorboardX import SummaryWriter

## helper function ##
def time_since(since):
  now = time.time()
  s = now - since
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def get_mutated_data(inps, color):
  res = []
  for a in inps:
    cnt = 0
    for col in color:
      if col in a:
        cnt += 1
    #if cnt == len(color):
    if cnt > 0:
      res.append(a)
  return res

def mutate_instruction(sentences, words, words_to_replace):
  sentences_mutate = []
  for sentence in sentences:
    sent = sentence
    for word, word_to_replace in zip(words, words_to_replace):
      sent = sent.replace(word, word_to_replace)
    sentences_mutate.append(sent)
  return sentences_mutate

def read_online_data(test_path, train_path, words, words_to_replace, number_train):
  inpss = []
  distinct_instr = set()
  with open(train_path, "r") as f:
    for line in f:
      inpss.append(line)
  inpss = get_mutated_data(inpss, words)

  for el in inpss:
    new_instr = el.split("\t")[1]
    distinct_instr.add(new_instr)
  inps_m = []
  targets_m = []
  instrs_m = []

  # randomize and sample equal data
  #inpsss = []
  #if len(words) > 1:
  each_d = 2
  distinct = []
  for el in list(distinct_instr):
    count = 0
    while count < each_d:
      elem = random.choice(inpss)
      if elem.split("\t")[1] == el:
        distinct.append(elem)
        count += 1
  tot_amnt = len(list(distinct_instr)) * each_d
  if number_train > tot_amnt:
    rest_amnt = len(list(distinct_instr)) * each_d
    additional = random.choices(inpss, k = rest_amnt)
    inpsss = distinct + additional
  else:
    inpsss = distinct
  #else:
    #inpsss = random.choices(inpss, k=number_train)

  for elem in inpsss:
    ar = elem.split("\t")
    inp = ar[0]
    ins = ar[1]
    lab = ar[2].replace("\n", "")
    inps_m.append(inp)
    targets_m.append(lab)
    instrs_m.append(ins)

  inps_rst = []
  targets_rst = []
  instrs_rst = []

  inpsss_rest = [i for i in inpss if not i in inpsss]
  for elem in inpsss_rest:
    ar = elem.split("\t")
    inp = ar[0]
    ins = ar[1]
    lab = ar[2].replace("\n", "")
    inps_rst.append(inp)
    targets_rst.append(lab)
    instrs_rst.append(ins)

  inpss = []
  with open(test_path, "r") as f:
    for line in f:
      inpss.append(line)
  inpss = get_mutated_data(inpss, words)

  inps_mt = []
  targets_mt = []
  instrs_mt = []
  for elem in inpss:
    ar = elem.split("\t")
    inp = ar[0]
    ins = ar[1]
    lab = ar[2].replace("\n", "")
    inps_mt.append(inp)
    targets_mt.append(lab)
    instrs_mt.append(ins)

  instrs_m = mutate_instruction(instrs_m, words, words_to_replace)
  instrs_mt = mutate_instruction(instrs_mt, words, words_to_replace)
  instrs_rst = mutate_instruction(instrs_rst, words, words_to_replace)

  return inps_m, instrs_m, targets_m, inps_rst, instrs_rst, targets_rst, inps_mt, instrs_mt, targets_mt

def read_merged_data(test_path, train_path, words, words_to_replace, number_train):
  inpss = []
  distinct_instr = set()
  with open(train_path, "r") as f:
    for line in f:
      inpss.append(line)
  with open(test_path, "r") as f:
    for line in f:
      inpss.append(line)
  inpss = get_mutated_data(inpss, words)

  for el in inpss:
    new_instr = el.split("\t")[1]
    distinct_instr.add(new_instr)
  print(list(distinct_instr))

  inps_m = []
  targets_m = []
  instrs_m = []

  # randomize and sample equal data
  inpsss = []
  #if len(words) > 1:
  each_d = 3
  distinct = []
  for el in list(distinct_instr):
    count = 0
    while count < each_d:
      elem = random.choice(inpss)
      if elem.split("\t")[1] == el:
        distinct.append(elem)
        count += 1
  tot_amnt = len(list(distinct_instr)) * each_d
  if number_train > tot_amnt:
    rest_amnt = len(list(distinct_instr)) * each_d
    additional = random.choices(inpss, k = rest_amnt)
    inpsss = distinct + additional
  else:
    inpsss = distinct
  #else:
    #inpsss = random.choices(inpss, k=number_train)

  for elem in inpsss:
    ar = elem.split("\t")
    inp = ar[0]
    ins = ar[1]
    lab = ar[2].replace("\n", "")
    inps_m.append(inp)
    targets_m.append(lab)
    instrs_m.append(ins)

  instrs_m = mutate_instruction(instrs_m, words, words_to_replace)

  return inps_m, instrs_m, targets_m

def instr_tensor_ext(all_words_comb, string):
  string = string.split(" ")
  tensor = torch.zeros(len(string)).long()
  for li, letter in enumerate(string):
    letter_index = all_words_comb.index(letter)
    tensor[li] = letter_index
  return tensor

def generate_batch_ext(dataset, start_index, len_example, len_labels, len_instr, all_words_comb,
                       batch_size, inps, instrs, labels):
  inp_tensor = torch.zeros(batch_size, len_example).long()
  lab_tensor = torch.zeros(batch_size, len_labels).long()
  ins_tensor = torch.zeros(batch_size, len_instr).long()
  for i in range(batch_size):
    inp = dataset.seq_tensor(inps[start_index + i])
    lab = dataset.categories_tensor(labels[start_index + i])
    ins = instr_tensor_ext(all_words_comb, instrs[start_index + i])
    inp_tensor[i, :] = inp
    lab_tensor[i, :] = lab
    ins_tensor[i, :] = ins
  # uncomment to do this with CPU
  # return Variable(inp_tensor), Variable(lab_tensor)
  return Variable(inp_tensor).cuda(), Variable(ins_tensor).cuda(), Variable(lab_tensor).cuda()

class EncoderExtended(nn.Module):
  def __init__(self, input_size, ext_input_size, hidden_size, n_layers=2):
    super(EncoderExtended, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.n_layers = n_layers

    self.embedding = nn.Embedding(input_size, hidden_size)
    self.embedding_ext = nn.Embedding(ext_input_size, hidden_size)
    #torch.nn.init.xavier_uniform(self.embedding_ext.weight)
    self.lamb = nn.Parameter(torch.rand(1))
    self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)

  def forward(self, inputs, hidden, batch_size, new_emb, lamb):
    # Note: we run this all at once (over the whole input sequence)
    if new_emb:
      embedded = self.embedding_ext(inputs)  # [1, 1, hidden_size]
      alpha_w = F.softmax(torch.mm(embedded.squeeze(1), self.embedding.weight.transpose(0, 1)),
                          dim=1).contiguous()  # [1, n_words_utterance]
      if lamb:
        embedded = self.lamb * embedded + (1 - self.lamb) * torch.mm(alpha_w, self.embedding.weight).unsqueeze(0) # [1, hidden_size]
    else:
      embedded = self.embedding(inputs)
    ht, hidden = self.lstm(embedded, hidden)  # [batch_size, seq_len, hidden_size]
    alph = 0
    if new_emb:
      alph = alpha_w
    return ht, hidden, alph

  def init_hidden(self, batch_size):
    h0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
    c0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
    return h0, c0

# training
def generate_position_ids(batch_size, len_targets):
  pos_tensor = torch.zeros(batch_size, len_targets).long()
  for i in range(batch_size):
    pos_tensor[i] = torch.LongTensor(range(0, len_targets))
  return Variable(pos_tensor).cuda()

def train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, n_hidden, batch_size, lamb,
              inp, instr, target, attn=False, eval=False):
  loss = 0
  enc_ext_optimizer.zero_grad()
  buffer_size = inp.size(0)
  for d in range(buffer_size):
    hid = enc_ext.init_hidden(1)
    cur_instr_len = instr.size(1)
    cntxt = Variable(torch.zeros(cur_instr_len, 1, 1, n_hidden)).cuda()
    for c in range(cur_instr_len):
        if instr[d,c].data[0] < dataset.n_words:
            ht, hid, _ = enc_ext(instr[d,c].unsqueeze(1), hid, 1, False, lamb)
        else:
            ht, hid, _ = enc_ext(instr[d,c].unsqueeze(1) % 20, hid, 1, True, lamb)
        cntxt[c] = ht.contiguous()
    cntxt = cntxt.squeeze(1).transpose(0,1)
    context = cntxt
    position_ids = generate_position_ids(batch_size, dataset.len_targets)
    if attn:
        output, vis_attn = decoder(inp[d], position_ids, batch_size, attn=True,
                             context=context)
        op = output.transpose(0,1) # seq_len, bs, class
        for c in range(dataset.len_targets):
            loss += criterion(op[c], target[d,c])
        #loss += criterion(output.view(batch_size, -1), target[:,c])
    else:
        output, _ = decoder(inp[d], position_ids, batch_size)
        op = output.transpose(0,1) # seq_len, bs, class
        for c in range(dataset.len_targets):
            loss += criterion(op[c], target[d,c])
    loss = loss / buffer_size

  if not eval:
    loss.backward()
    enc_ext_optimizer.step()

  return loss.data[0] / dataset.len_targets

def accuracy_test_data_ext(dataset, enc_ext, decoder, inps_mt, instrs_mt, targets_mt, all_words_comb, n_hidden, batch_size, lamb,
                           attn=False):
  it = len(inps_mt) / batch_size
  acc_tot = 0
  acc_tot_seq = 0
  for i in range(int(it)):
    decoder.eval()
    enc_ext.eval()
    start_index = i * batch_size
    # dataset, start_index, len_example, len_labels, len_instr, batch_size, inps, instrs, labels
    inp, instr, target = generate_batch_ext(dataset, start_index, dataset.len_example, dataset.len_targets,
                                            dataset.len_instr, all_words_comb, batch_size, inps_mt, instrs_mt, targets_mt)
    hid = enc_ext.init_hidden(batch_size)
    cntxt = Variable(torch.zeros(dataset.len_instr, batch_size, 1, n_hidden)).cuda()
    for c in range(dataset.len_instr):
      if instr[:, c].data[0] < dataset.n_words:
        ht, hid, _ = enc_ext(instr[:, c].unsqueeze(1), hid, batch_size, False, lamb)
      else:
        ht, hid, _ = enc_ext(instr[:, c].unsqueeze(1) % 20, hid, batch_size, True, lamb)
      cntxt[c] = ht.contiguous()
    cntxt = cntxt.squeeze(2).transpose(0, 1).contiguous()
    context = cntxt
    # ht, hid = enc_ext(instr, hid, batch_size, False)
    # context = ht
    position_ids = generate_position_ids(batch_size, dataset.len_targets)
    if attn:
      pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      output, vis_attn = decoder(inp, position_ids, batch_size, attn=True,
                                 context=context)
      op = output.transpose(0, 1)  # seq_len, bs, class
      for c in range(dataset.len_targets):
        tgt_seq[c] = target[:, c]
        pred_seq[c] = op[c].max(1)[1]
        accuracy = (op[c].max(1)[1] == target[:, c]).float().sum().float() / batch_size
        acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * 23
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
      # print((pred_seq == tgt_seq).float().sum(dim=0))
    else:
      pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      output, _ = decoder(inp, position_ids, batch_size)
      op = output.transpose(0, 1)  # seq_len, bs, class
      for c in range(dataset.len_targets):
        tgt_seq[c] = target[:, c]
        pred_seq[c] = op[c].max(1)[1]
        accuracy = (op[c].max(1)[1] == target[:, c]).float().sum().float() / batch_size
        acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * 23
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
      # print((pred_seq == tgt_seq).float().sum(dim=0))
  return acc_tot / (it * dataset.len_targets), acc_tot_seq / it