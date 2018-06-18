import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random

def generate_position_ids_word(batch_size, len_instr):
  pos_tensor = torch.zeros(batch_size, len_instr).long()
  for i in range(batch_size):
    pos_tensor[i] = torch.LongTensor(range(0, len_instr))
  return Variable(pos_tensor).cuda()

def generate_position_ids(batch_size, len_example):
  pos_tensor = torch.zeros(batch_size, len_example).long()
  for i in range(batch_size):
    pos_tensor[i] = torch.LongTensor(range(0, len_example))
  return Variable(pos_tensor).cuda()

# training
def train(dataset, encoder, decoder, enc_optimizer, optimizer, criterion, len_targets, batch_size,
          inp, instr, target, attn=False):
  loss = 0
  optimizer.zero_grad()
  enc_optimizer.zero_grad()
  pos_id_words = generate_position_ids_word(batch_size, dataset.len_instr)
  encoder_ht = encoder(instr, pos_id_words, batch_size)
  context = encoder_ht
  position_ids = generate_position_ids(batch_size, len_targets)
  if attn:
    output, vis_attn = decoder(inp, position_ids, batch_size, attn=True,
                               context=context)
    op = output.transpose(0, 1)  # seq_len, bs, class
    for c in range(len_targets):
      loss += criterion(op[c], target[:, c])
      # loss += criterion(output.view(batch_size, -1), target[:,c])
  else:
    output, _ = decoder(inp, position_ids, batch_size)
    op = output.transpose(0, 1)  # seq_len, bs, class
    for c in range(len_targets):
      loss += criterion(op[c], target[:, c])

  loss.backward()
  optimizer.step()
  enc_optimizer.step()

  return loss.data[0] / len_targets