import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random

def generate_position_ids(batch_size, len_targets):
  pos_tensor = torch.zeros(batch_size, len_targets).long()
  for i in range(batch_size):
    pos_tensor[i] = torch.LongTensor(range(0, len_targets))
  return Variable(pos_tensor).cuda()

# training
def train(encoder, decoder, enc_optimizer, optimizer, criterion, len_targets, batch_size,
          inp, instr, target, attn=False):
  loss = 0
  optimizer.zero_grad()
  enc_optimizer.zero_grad()
  encoder_hidden = encoder.init_hidden(batch_size)
  encoder_ht, encoder_hidden = encoder(instr, encoder_hidden, batch_size)
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