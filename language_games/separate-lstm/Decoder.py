import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, n_layers=2, dropout_p=0.2,
               example_len=30, ponder_step=5, combine=True, dot=True):
    super(Decoder, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.input_dropout = nn.Dropout(p=dropout_p)
    self.embed = nn.Embedding(input_size, hidden_size)
    if combine:
      self.lstm = nn.LSTM(2 * hidden_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
    else:
      self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
    if dot:
      self.output = nn.Linear(hidden_size, output_size)
    else:
      self.output = nn.Linear(2 * hidden_size, output_size)
    self.ponder_step = ponder_step
    self.example_len = example_len
    self.combine = combine
    self.dot = dot

    # for context-attended output
    self.linear_out = nn.Linear(hidden_size * 2, hidden_size)
    # for mapping to probability
    self.linear_ponder = nn.Linear(hidden_size, 1)
    self.ponder_noise = nn.Embedding(ponder_step, hidden_size)
    self.mlp = nn.Linear(2*hidden_size, 1)

  def forward(self, input, hidden, batch_size, attn=False, context=None, prev_ht=None, ponder=False):
    embedded = self.embed(input)  # [batch_size, seq_len, embed_size]
    embedded = self.input_dropout(embedded)
    inp_embedded = embedded
    # for visualization #
    vis_attn = Variable(torch.zeros(self.ponder_step, batch_size, 1
                                    , self.example_len)).cuda()

    if not attn:
      seq_len = embedded.size(1)
      prev_ht = self.init_prev_ht(batch_size, seq_len)
      if self.combine:
        embedded = torch.cat((embedded, prev_ht), dim=2)
      ht, hidden = self.lstm(embedded, hidden)  # [batch_size, seq_len, hidden_size]
      if not self.dot:
        output = None
      else:
        output = F.log_softmax(self.output(ht.squeeze(1)), dim=1)
      out_ht = ht
    else:
      ### pondering step ###
      if ponder:
        pondered_ht = Variable(torch.zeros(self.ponder_step, batch_size, 1, self.hidden_size)).cuda()
        pondered_prob_ht = Variable(torch.zeros(self.ponder_step, batch_size, 1, 1)).cuda()
        pondered_hidden = Variable(torch.zeros(self.ponder_step, self.n_layers, batch_size
                                               , self.hidden_size)).cuda()
        pondered_cell = Variable(torch.zeros(self.ponder_step, self.n_layers, batch_size
                                             , self.hidden_size)).cuda()

        for c in range(0, self.ponder_step):
          noise = Variable((c) * torch.ones(batch_size, 1).long()).cuda()
          noise_embed = self.ponder_noise(noise)  # [batch_size, 1, n_hidden]
          embedded = torch.add(inp_embedded, noise_embed)
          embedded = torch.cat((embedded, prev_ht), dim=2)
          ht, hidden = self.lstm(embedded, hidden)
          prob_ht = self.linear_ponder(ht)
          # calulate N attn
          attn = torch.bmm(ht, context.transpose(1, 2))
          attn = F.softmax(attn.view(-1, self.example_len), dim=1).view(batch_size, -1, self.example_len)

          # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
          mix = torch.bmm(attn, context)

          # concat -> (batch, out_len, 2*dim)
          combined = torch.cat((mix, ht), dim=2)
          # output -> (batch, out_len, dim)
          ht = F.tanh(self.linear_out(combined.view(-1, 2 * self.hidden_size))).view(
            batch_size, -1, self.hidden_size)
          # gather the pondered hidden states and its probability
          pondered_ht[c] = ht
          pondered_prob_ht[c] = prob_ht
          pondered_hidden[c] = hidden[0]
          pondered_cell[c] = hidden[1]
          vis_attn[c] = attn

        pondered_prob_ht = F.softmax(pondered_prob_ht, dim=0)
        ht = torch.mul(pondered_ht, pondered_prob_ht)
        ht = ht.sum(0).view(batch_size, 1, -1)
        pondered_prob_hid = pondered_prob_ht.view(self.ponder_step, 1, batch_size, 1)
        hid = torch.mul(pondered_hidden, pondered_prob_hid)
        hid = hid.sum(0).view(self.n_layers, batch_size, -1)
        cell = torch.mul(pondered_cell, pondered_prob_hid)
        cell = cell.sum(0).view(self.n_layers, batch_size, -1)
        hidden = (hid, cell)

      ### attention with dot mechanism Luong ###
      else:
        if self.combine:
          embedded = torch.cat((embedded, prev_ht), dim=2)
        ht, hidden = self.lstm(embedded, hidden)  # [batch_size, 1, hidden_size]
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        # in_len is the number of characters in total k-factors
        # out_len is the number of step which is 1
        if self.dot:
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
        else:
          # expand decoder states and transpose
          ht_exp = ht.expand(batch_size, self.example_len, self.hidden_size)
          ht_tr = ht_exp.contiguous().view(-1, self.hidden_size)

          # reshape encoder states to allow batchwise computation
          context_tr = context.contiguous().view(-1, self.hidden_size)

          mlp_input = torch.cat((ht_tr, context_tr), dim=1)

          # apply mlp and respape to get in correct form
          mlp_output = self.mlp(mlp_input)
          attn = mlp_output.view(batch_size, 1, self.example_len)
          attn = F.softmax(attn.view(-1, self.example_len), dim=1).view(batch_size, -1, self.example_len)
          # (batch, out_len, dim)
          mix = torch.bmm(attn, context)
          out_ht = torch.cat((mix, ht), dim=2)

      output = F.log_softmax(self.output(out_ht.squeeze(1)), dim=1)

    return output, ht, hidden, vis_attn

  def init_hidden(self, batch_size):
    h0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
    c0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
    return h0, c0

  def init_prev_ht(self, batch_size, seq_len):
    ht = Variable(torch.zeros(batch_size, seq_len, self.hidden_size)).cuda()
    return ht