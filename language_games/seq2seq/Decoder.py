import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, n_layers=2, dropout_p=0.2, example_len=15,
               concat=False):
    super(Decoder, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.input_dropout = nn.Dropout(p=dropout_p)
    self.embed = nn.Embedding(input_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
    if concat:
      self.output = nn.Linear(2 * hidden_size, output_size)
    else:
      self.output = nn.Linear(hidden_size, output_size)
    self.example_len = example_len
    self.concat = concat

    # for context-attended output
    self.linear_out = nn.Linear(hidden_size * 2, hidden_size)
    # for combining utterance context and init block context
    self.combine = nn.Linear(hidden_size * 2, hidden_size)
    self.mlp = nn.Linear(2 * hidden_size, 1)

  def forward(self, inputs, hidden, batch_size, attn=False, context=None):
    embedded = self.embed(inputs)  # [batch_size, seq_len, embed_size]
    embedded = self.input_dropout(embedded)
    inp_embedded = embedded
    output = None
    # for visualization #
    vis_attn = Variable(torch.zeros(1, batch_size, 1, self.example_len)).cuda()

    if not attn:
      ht, hidden = self.lstm(embedded, hidden)  # [batch_size, seq_len, hidden_size]
      output = F.log_softmax(self.output(ht.squeeze(1)), dim=1)
      out_ht = ht
    else:
      ### attention with mlp concat bahdanau ###
      ht, hidden = self.lstm(embedded, hidden)  # [batch_size, 1, hidden_size]
      # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
      # in_len is the number of blocks in initial block configuration
      # out_len is the number of step which is 1
      if self.concat:
        ht_exp = ht.expand(batch_size, self.example_len, self.hidden_size)
        ht_tr = ht_exp.contiguous().view(-1, self.hidden_size)

        # reshape encoder states to allow batchwise computation
        context_tr = context.contiguous().view(-1, self.hidden_size)

        mlp_input = torch.cat((ht_tr, context_tr), dim=1)

        # apply mlp and respape to get in correct form
        mlp_output = self.mlp(mlp_input)
        attn = mlp_output.view(batch_size, 1, self.example_len)
        attn = F.softmax(attn.view(-1, self.example_len), dim=1).view(batch_size, -1, self.example_len)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        out_ht = torch.cat((mix, ht), dim=2)

        vis_attn[0] = attn
      else:
        attn = torch.bmm(ht, context.transpose(1, 2))
        attn = F.softmax(attn.view(-1, self.example_len), dim=1).view(batch_size, -1, self.example_len)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, ht), dim=2)
        # output -> (batch, out_len, dim)
        ht = F.tanh(self.linear_out(combined.view(-1, 2 * self.hidden_size))).view(
          batch_size, -1, self.hidden_size)
        vis_attn[0] = attn
        out_ht = ht

      output = F.log_softmax(self.output(out_ht.squeeze(1)), dim=1)

    return output, ht, hidden, vis_attn

  def init_hidden(self, batch_size):
    h0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
    c0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
    return h0, c0

  def init_prev_ht(self, batch_size, seq_len):
    ht = Variable(torch.zeros(batch_size, seq_len, self.hidden_size)).cuda()
    return ht