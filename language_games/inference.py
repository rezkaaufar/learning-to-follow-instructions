import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random

def generate_position_ids(batch_size, len_targets):
  pos_tensor = torch.zeros(batch_size, len_targets).long()
  for i in range(batch_size):
    pos_tensor[i] = torch.LongTensor(range(0, len_targets))
  return Variable(pos_tensor).cuda()

def convert_to_characters(dataset, arr):
  res = ""
  for ar in arr:
    res += dataset.all_characters[int(ar)] + " "
  return res

def infer(dataset, encoder, decoder, inps_t, instrs_t, targets_t, batch_size, attn=True):
  acc_tot_seq = 0
  i = 0
  decoder.eval()
  encoder.eval()
  start_index = i * batch_size
  inp, instr, target = dataset.generate_batch(start_index, batch_size, inps_t, instrs_t, targets_t)
  encoder_hidden = encoder.init_hidden(batch_size)
  encoder_ht, encoder_hidden = encoder(instr, encoder_hidden, batch_size)
  context = encoder_ht
  position_ids = generate_position_ids(batch_size, dataset.len_targets)
  if attn:
    pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
    tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
    output, vis_attn = decoder(inp, position_ids, batch_size, dataset.len_example, dataset.len_targets,
                               dataset.len_instr, attn=True,
                               context=context)
    op = output.transpose(0, 1)  # seq_len, bs, class
    for c in range(dataset.len_targets):
      tgt_seq[c] = target[:, c]
      pred_seq[c] = op[c].max(1)[1]
      accuracy = (op[c].max(1)[1] == target[:, c]).float().sum().float() / batch_size
      #acc_tot += accuracy.data[0]
    truth = Variable(torch.ones(batch_size)).cuda() * 23
    acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
    print(acc_tot_seq)
    for c in range(batch_size):
      #print(inps_t[start_index + c], instrs_t[start_index + c])
      print(convert_to_characters(dataset, pred_seq[:, c].data) == convert_to_characters(dataset, tgt_seq[:, c].data))
      # print(convert_to_characters(dataset, tgt_seq[:, c].data))
      # print(convert_to_characters(dataset, pred_seq[:, c].data))
    #acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size