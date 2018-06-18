import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random

def convert_to_characters(dataset, arr):
  res = ""
  for ar in arr:
    res += dataset.all_characters[int(ar)] + " "
  return res

def infer(dataset, encoder, decoder, inps_t, instrs_t, targets_t, batch_size, attn=True):
  start_index = random.choice(len(inps_t))
  acc_tot = 0
  acc_tot_seq = 0
  decoder.zero_grad()
  encoder.zero_grad()
  decoder.eval()
  encoder.eval()
  inp, instr, target = dataset.generate_batch(start_index, batch_size, inps_t, instrs_t, targets_t)
  encoder_hidden = encoder.init_hidden(batch_size)
  encoder_ht, encoder_hidden = encoder(instr, encoder_hidden, batch_size)
  context = encoder_ht
  hidden = encoder_hidden
  if attn:
    pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
    tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
    for c in range(dataset.len_targets):
      output, ht, hidden, vis_attn = decoder(inp[:, c].unsqueeze(1), hidden, batch_size, attn=True,
                                             context=context)
      tgt_seq[c] = target[:, c]
      pred_seq[c] = output.max(1)[1]
      accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
      acc_tot += accuracy.data[0]
    truth = Variable(torch.ones(batch_size)).cuda() * 23
    # print(pred_seq, tgt_seq)
    for c in range(batch_size):
      print(inps_t[start_index + c], instrs_t[start_index + c])
      print(convert_to_characters(tgt_seq[:, c].data))
      print(convert_to_characters(pred_seq[:, c].data))
    acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size