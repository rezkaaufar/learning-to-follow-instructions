import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random

def generate_position_ids_word(batch_size, len_instr):
  pos_tensor = torch.zeros(batch_size, len_instr).long()
  for i in range(batch_size):
    pos_tensor[i] = torch.LongTensor(range(0, len_instr))
  return Variable(pos_tensor).cuda()

# training
def train(encoder, decoder, enc_optimizer, optimizer, criterion, len_targets, len_instr, batch_size,
          inp, instr, target, attn=False):
  loss = 0
  optimizer.zero_grad()
  enc_optimizer.zero_grad()
  pos_id_words = generate_position_ids_word(batch_size, len_instr)
  encoder_ht = encoder(instr, pos_id_words, batch_size)
  context = encoder_ht
  hidden = decoder.init_hidden(batch_size)
  if attn:
    ht = decoder.init_prev_ht(batch_size, 1)
    for c in range(len_targets):
      output, ht, hidden, vis_attn = decoder(inp[:, c].unsqueeze(1), hidden, batch_size, attn=True,
                                             context=context)
      loss += criterion(output.view(batch_size, -1), target[:, c])
  else:
    for c in range(len_targets):
      output, ht, hidden, _ = decoder(inp[:, c].unsqueeze(1), hidden, batch_size)
      loss += criterion(output.view(batch_size, -1), target[:, c])

  loss.backward()
  torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)
  optimizer.step()

  return loss.data[0] / len_targets