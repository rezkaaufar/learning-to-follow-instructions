import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, n_layers=2, bi=False):
    super(Encoder, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.n_layers = n_layers

    self.embedding = nn.Embedding(input_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, bidirectional=bi)
    self.bi = bi

  def forward(self, inputs, hidden, batch_size):
    # Note: we run this all at once (over the whole input sequence)
    embedded = self.embedding(inputs)
    ht, hidden = self.lstm(embedded, hidden)  # [batch_size, seq_len, hidden_size]
    return ht, hidden

  def init_hidden(self, batch_size):
    if self.bi:
      h0 = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size)).cuda()
      c0 = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size)).cuda()
    else:
      h0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
      c0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
    return h0, c0