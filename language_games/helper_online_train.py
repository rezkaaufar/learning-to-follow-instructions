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

def write_merged_data(test_path, train_path, words, words_to_replace, number_train):
  name = "_".join(str(z) for z in words)
  fw = open("./dataset/online_test/all.txt", "w")
  #fw = open("./dataset/online_test/" + name + ".txt", "w")
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

  for i in range(len(inps_m)):
    fw.write(inps_m[i] + "\t" + instrs_m[i] + "\t" + targets_m[i])
    fw.write("\n")
  fw.close()
  #return inps_m, instrs_m, targets_m

# which_data = "utter_blocks"
# words_to_replace = ["add", "red", "orange", "1st", "3rd", "5th", "even", "leftmost"]
# words_replacement = ["et", "roze", "oranje", "1", "3", "5", "ev", "lftmst"]
# wtr = [["remove", "brown", "cyan", "2nd", "4th", "6th", "odd", "every", "rightmost", "at", "to", "tile"]]
# wrs = [["rmv", "braun", "cyaan", "2", "4", "6", "ot", "evr", "rms", "di", "ke", "sqr"]]
# conf = [["remove"], ["brown", "cyan"], ["2nd", "4th", "6th", "odd", "every"]]
# conf_rep = [["rmv"], ["braun", "cyaan"], ["2", "4", "6", "ot", "evr"]]
# config = list(itertools.product(*conf))
# config_rep = list(itertools.product(*conf_rep))
# number_train = 20
# for el, el_rep in zip(wtr, wrs):
#   write_merged_data("dataset/lang_games_data_artificial_test_nvl_" + which_data + "_50000.txt",
#                     "dataset/lang_games_data_artificial_valid_nvl_" + which_data + "_50000.txt",
#                      el, el_rep, number_train)

def read_merged_data(online_path):
  inpss = []
  with open(online_path, "r") as f:
    for line in f:
      inpss.append(line)
  inps_m, targets_m, instrs_m = [], [], []
  for elem in inpss:
    ar = elem.split("\t")
    inp = ar[0]
    ins = ar[1]
    lab = ar[2].replace("\n", "")
    inps_m.append(inp)
    targets_m.append(lab)
    instrs_m.append(ins)
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
  # return Variable(inp_tensor), Variable(ins_tensor), Variable(lab_tensor)
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

def train_ext_unfreezed(enc_ext, decoder, enc_ext_optimizer, decoder_optimizer, criterion, dataset, len_ex, len_tgt, len_ins,
              n_hidden, batch_size, lamb,
              inp, instr, target, attn=False, eval=False):
  loss = 0
  enc_ext_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
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
    position_ids = generate_position_ids(batch_size, len_tgt)
    if attn:
        output, vis_attn = decoder(inp[d], position_ids, batch_size, len_ex, len_tgt, len_ins, attn=True,
                             context=context)
        op = output.transpose(0,1) # seq_len, bs, class
        for c in range(len_tgt):
            loss += criterion(op[c], target[d,c])
        #loss += criterion(output.view(batch_size, -1), target[:,c])
    else:
        output, _ = decoder(inp[d], position_ids, batch_size)
        op = output.transpose(0,1) # seq_len, bs, class
        for c in range(len_tgt):
            loss += criterion(op[c], target[d,c])
    loss = loss / buffer_size

  if not eval:
    loss.backward()
    enc_ext_optimizer.step()
    decoder_optimizer.step()

  return loss.data[0] / len_tgt

def train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, len_ex, len_tgt, len_ins,
              n_hidden, batch_size, lamb,
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
    position_ids = generate_position_ids(batch_size, len_tgt)
    if attn:
        output, vis_attn = decoder(inp[d], position_ids, batch_size, len_ex, len_tgt, len_ins, attn=True,
                             context=context)
        op = output.transpose(0,1) # seq_len, bs, class
        for c in range(len_tgt):
            loss += criterion(op[c], target[d,c])
        #loss += criterion(output.view(batch_size, -1), target[:,c])
    else:
        output, _ = decoder(inp[d], position_ids, batch_size)
        op = output.transpose(0,1) # seq_len, bs, class
        for c in range(len_tgt):
            loss += criterion(op[c], target[d,c])
    loss = loss / buffer_size

  if not eval:
    loss.backward()
    enc_ext_optimizer.step()

  return loss.data[0] / len_tgt

def accuracy_test_data_ext(dataset, len_ex, len_tgt, len_ins, enc_ext, decoder,
                           inps_mt, instrs_mt, targets_mt, all_words_comb, n_hidden, batch_size, lamb,
                           attn=False):
  it = len(inps_mt) / batch_size
  acc_tot = 0
  acc_tot_seq = 0
  for i in range(int(it)):
    decoder.eval()
    enc_ext.eval()
    start_index = i * batch_size
    # dataset, start_index, len_example, len_labels, len_instr, batch_size, inps, instrs, labels
    inp, instr, target = generate_batch_ext(dataset, start_index, len_ex, len_tgt,
                                            len_ins, all_words_comb, batch_size, inps_mt, instrs_mt, targets_mt)
    hid = enc_ext.init_hidden(batch_size)
    cntxt = Variable(torch.zeros(len_ins, batch_size, 1, n_hidden)).cuda()
    for c in range(len_ins):
      if instr[:, c].data[0] < dataset.n_words:
        ht, hid, _ = enc_ext(instr[:, c].unsqueeze(1), hid, batch_size, False, lamb)
      else:
        ht, hid, _ = enc_ext(instr[:, c].unsqueeze(1) % 20, hid, batch_size, True, lamb)
      cntxt[c] = ht.contiguous()
    cntxt = cntxt.squeeze(2).transpose(0, 1).contiguous()
    context = cntxt
    # ht, hid = enc_ext(instr, hid, batch_size, False)
    # context = ht
    position_ids = generate_position_ids(batch_size, len_tgt)
    if attn:
      pred_seq = Variable(torch.zeros(len_tgt, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(len_tgt, batch_size)).cuda()
      output, vis_attn = decoder(inp, position_ids, batch_size, len_ex, len_tgt, len_ins, attn=True,
                                 context=context)
      op = output.transpose(0, 1)  # seq_len, bs, class
      for c in range(len_tgt):
        tgt_seq[c] = target[:, c]
        pred_seq[c] = op[c].max(1)[1]
        accuracy = (op[c].max(1)[1] == target[:, c]).float().sum().float() / batch_size
        acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * len_tgt
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
      # print((pred_seq == tgt_seq).float().sum(dim=0))
    else:
      pred_seq = Variable(torch.zeros(len_tgt, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(len_tgt, batch_size)).cuda()
      output, _ = decoder(inp, position_ids, batch_size)
      op = output.transpose(0, 1)  # seq_len, bs, class
      for c in range(len_tgt):
        tgt_seq[c] = target[:, c]
        pred_seq[c] = op[c].max(1)[1]
        accuracy = (op[c].max(1)[1] == target[:, c]).float().sum().float() / batch_size
        acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * len_tgt
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
      # print((pred_seq == tgt_seq).float().sum(dim=0))
  return acc_tot / (it * len_tgt), acc_tot_seq / it

### helper for train attn ###
class Encoder2Extended(nn.Module):
  def __init__(self, input_size, ext_input_size, hidden_size):
    super(Encoder2Extended, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size

    self.embedding = nn.Embedding(input_size, hidden_size)
    self.embedding_ext = nn.Embedding(ext_input_size, hidden_size)
    self.attention_1 = nn.Parameter(torch.rand(1, hidden_size))
    self.attention_2 = nn.Parameter(torch.rand(1, hidden_size))
    self.attention_3 = nn.Parameter(torch.rand(1, hidden_size))

  def forward(self, inputs, batch_size, attn, new_emb):
    # Note: we run this all at once (over the whole input sequence)
    if new_emb:
      embedded = self.embedding_ext(inputs)  # [batch_size, seq_len, hidden_size]
    else:
      embedded = self.embedding(inputs)  # [batch_size, seq_len, hidden_size]
    pos_index = torch.LongTensor([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 17]).cuda()
    col_index = torch.LongTensor([8, 9, 14, 15]).cuda()
    com_index = torch.LongTensor([6, 16]).cuda()
    if not attn:
      alpha_1 = torch.bmm(embedded, self.attention_1.unsqueeze(0).expand(batch_size, -1, -1).transpose(1,2))
      result_1 = torch.sum(alpha_1 * embedded, dim=1)
      alpha_2 = torch.bmm(embedded, self.attention_2.unsqueeze(0).expand(batch_size, -1, -1).transpose(1,2))
      result_2 = torch.sum(alpha_2 * embedded, dim=1)
      alpha_3 = torch.bmm(embedded, self.attention_3.unsqueeze(0).expand(batch_size, -1, -1).transpose(1,2))
      result_3 = torch.sum(alpha_3 * embedded, dim=1)
    else:
      result_1 = torch.sum(self.embedding.weight[(com_index)].mean(0) * embedded, dim=1)
      result_2 = torch.sum(self.embedding.weight[(col_index)].mean(0) * embedded, dim=1)
      result_3 = torch.sum(self.embedding.weight[(pos_index)].mean(0) * embedded, dim=1)
    return torch.cat([result_1.unsqueeze(1), result_2.unsqueeze(1), result_3.unsqueeze(1)], dim=1)

# training
def train_ext_2(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, len_ex, len_tgt, len_ins,
              n_hidden, batch_size, inp, instr, target, mean_attn, attn=False, eval=False):
  loss = 0
  enc_ext_optimizer.zero_grad()
  command_len = 3

  cntxt = Variable(torch.zeros(len_ins, 1, 3, n_hidden)).cuda()
  for c in range(len_ins):
    if instr[0, c].data[0] < dataset.n_words:
      ht = enc_ext(instr[0, c].unsqueeze(1), 1, mean_attn, False)
    else:
      ht = enc_ext(instr[0, c].unsqueeze(1) % 20, 1, mean_attn, True)
    cntxt[c] = ht.contiguous()
  cntxt = cntxt.squeeze(1).transpose(0, 1)

  context = torch.sum(cntxt, dim=1).unsqueeze(0)
  # context = cntxt
  position_ids = generate_position_ids(batch_size, len_ex)
  if attn:
    output, vis_attn = decoder(inp, position_ids, batch_size, len_ex, len_tgt, command_len, attn=True,
                               context=context)
    op = output.transpose(0, 1)  # seq_len, bs, class
    for c in range(len_ex):
      loss += criterion(op[c], target[:, c])
      # loss += criterion(output.view(batch_size, -1), target[:,c])
  else:
    output, _ = decoder(inp, position_ids, batch_size, len_ex, len_ins, len_tgt)
    op = output.transpose(0, 1)  # seq_len, bs, class
    for c in range(len_ex):
      loss += criterion(op[c], target[:, c])
  if not eval:
    loss.backward()
    enc_ext_optimizer.step()

  return loss.data[0] / len_tgt

def train_ext_2_unfreezed(enc_ext, decoder, enc_ext_optimizer, decoder_optimizer, criterion, dataset, len_ex, len_tgt, len_ins,
              n_hidden, batch_size, inp, instr, target, mean_attn, attn=False, eval=False):
  loss = 0
  enc_ext_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  command_len = 3

  cntxt = Variable(torch.zeros(len_ins, 1, 3, n_hidden)).cuda()
  for c in range(len_ins):
    if instr[0, c].data[0] < dataset.n_words:
      ht = enc_ext(instr[0, c].unsqueeze(1), 1, mean_attn, False)
    else:
      ht = enc_ext(instr[0, c].unsqueeze(1) % 20, 1, mean_attn, True)
    cntxt[c] = ht.contiguous()
  cntxt = cntxt.squeeze(1).transpose(0, 1)

  context = torch.sum(cntxt, dim=1).unsqueeze(0)
  # context = cntxt
  position_ids = generate_position_ids(batch_size, len_ex)
  if attn:
    output, vis_attn = decoder(inp, position_ids, batch_size, len_ex, len_tgt, command_len, attn=True,
                               context=context)
    op = output.transpose(0, 1)  # seq_len, bs, class
    for c in range(len_ex):
      loss += criterion(op[c], target[:, c])
      # loss += criterion(output.view(batch_size, -1), target[:,c])
  else:
    output, _ = decoder(inp, position_ids, batch_size, len_ex, len_ins, len_tgt)
    op = output.transpose(0, 1)  # seq_len, bs, class
    for c in range(len_ex):
      loss += criterion(op[c], target[:, c])
  if not eval:
    loss.backward()
    enc_ext_optimizer.step()
    decoder_optimizer.zero_grad()

  return loss.data[0] / len_tgt

def accuracy_test_data_ext_2(dataset, len_ex, len_tgt, len_ins, enc_ext, decoder,
                           inps_mt, instrs_mt, targets_mt, all_words_comb, n_hidden, batch_size,
                           mean_attn, attn=False):
  it = len(inps_mt) / batch_size
  acc_tot = 0
  acc_tot_seq = 0
  command_len = 3
  for i in range(int(it)):
    decoder.eval()
    enc_ext.eval()
    start_index = i * batch_size
    # len_ex = len(inps_mt[start_index].split(" "))
    # len_ins = len(instrs_mt[start_index].split(" "))
    # len_tgt = len(targets_mt[start_index].split(" "))
    inp, instr, target = generate_batch_ext(dataset, start_index, len_ex, len_tgt, len_ins, all_words_comb,
                                            batch_size, inps_mt, instrs_mt, targets_mt)
    cntxt = Variable(torch.zeros(len_ins, batch_size, 3, n_hidden)).cuda()
    for c in range(len_ins):
      if instr[:, c].data[0] < dataset.n_words:
        ht = enc_ext(instr[:, c].unsqueeze(1), batch_size, mean_attn, False)
      else:
        ht = enc_ext(instr[:, c].unsqueeze(1) % 20, batch_size, mean_attn, True)
      cntxt[c] = ht.contiguous()
    cntxt = cntxt.squeeze(1).transpose(0, 1)
    context = torch.sum(cntxt, dim=1).unsqueeze(0)
    # ht, hid = enc_ext(instr, hid, batch_size, False)
    # context = ht
    position_ids = generate_position_ids(batch_size, len_ex)
    if attn:
      pred_seq = Variable(torch.zeros(len_tgt, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(len_tgt, batch_size)).cuda()
      output, vis_attn = decoder(inp, position_ids, batch_size, len_ex, len_tgt, command_len, attn=True,
                                 context=context)
      op = output.transpose(0, 1)  # seq_len, bs, class
      for c in range(len_tgt):
        tgt_seq[c] = target[:, c]
        pred_seq[c] = op[c].max(1)[1]
        accuracy = (op[c].max(1)[1] == target[:, c]).float().sum().float() / batch_size
        acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * len_tgt
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
      # print((pred_seq == tgt_seq).float().sum(dim=0))
    else:
      pred_seq = Variable(torch.zeros(len_tgt, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(len_tgt, batch_size)).cuda()
      output, _ = decoder(inp, position_ids, batch_size)
      op = output.transpose(0, 1)  # seq_len, bs, class
      for c in range(len_tgt):
        tgt_seq[c] = target[:, c]
        pred_seq[c] = op[c].max(1)[1]
        accuracy = (op[c].max(1)[1] == target[:, c]).float().sum().float() / batch_size
        acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * len_tgt
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
      # print((pred_seq == tgt_seq).float().sum(dim=0))
  return acc_tot / (it * len_tgt), acc_tot_seq / it