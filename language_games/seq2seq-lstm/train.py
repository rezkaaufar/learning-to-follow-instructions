import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random

# training
def train(encoder, decoder, enc_optimizer, optimizer, criterion, len_targets, batch_size,
          inp, instr, target, attn=False):
  loss = 0
  enc_optimizer.zero_grad()
  optimizer.zero_grad()
  encoder_hidden = encoder.init_hidden(batch_size)
  encoder_ht, encoder_hidden = encoder(instr, encoder_hidden, batch_size)
  context = encoder_ht
  hidden = encoder_hidden
  if attn:
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
  torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
  enc_optimizer.step()
  optimizer.step()

  return loss.data[0] / len_targets