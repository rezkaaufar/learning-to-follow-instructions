import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, n_layers=2, dropout_p=0.2,
               grammar_len=15, n_k_factors=5):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.input_dropout = nn.Dropout(p=dropout_p)
    self.embed = nn.Embedding(input_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
    self.output = nn.Linear(hidden_size, output_size)
    self.n_k_factors = n_k_factors
    self.grammar_len = grammar_len

    # for context-attended output
    self.linear_out = nn.Linear(hidden_size * 2, hidden_size)
    # for mapping to probability
    self.linear_ponder = nn.Linear(hidden_size, 1)
    self.ponder_noise = nn.Embedding(n_k_factors, hidden_size)

  def forward(self, input, hidden, batch_size, attn=False, context=None, prev_ht=None, ponder=False):
    embedded = self.embed(input)  # [batch_size, seq_len, embed_size]
    embedded = self.input_dropout(embedded)
    inp_embedded = embedded
    # for visualization #
    vis_attn = Variable(torch.zeros(self.n_k_factors, batch_size, 1
                                           , self.grammar_len)).cuda()

    if not attn:
      seq_len = embedded.size(1)
      #prev_ht = self.init_prev_ht(batch_size, seq_len)
      #embedded = torch.cat((embedded, prev_ht), dim=2)
      ht, hidden = self.lstm(embedded, hidden)  # [batch_size, seq_len, hidden_size]
      output = self.output(ht.squeeze(1))
    else:
      ### pondering step ###
      if ponder:
        pondered_ht = Variable(torch.zeros(self.n_k_factors, batch_size, 1, self.hidden_size)).cuda()
        pondered_prob_ht = Variable(torch.zeros(self.n_k_factors, batch_size, 1, 1)).cuda()
        pondered_hidden = Variable(torch.zeros(self.n_k_factors, self.n_layers, batch_size
                                               , self.hidden_size)).cuda()
        pondered_cell = Variable(torch.zeros(self.n_k_factors, self.n_layers, batch_size
                                             , self.hidden_size)).cuda()
        # for testing #
        # noise = Variable((0) * torch.ones(batch_size, 1).long()).cuda()
        # noise_embed = self.ponder_noise(noise)  # [batch_size, 1, n_hidden]
        # embedded = torch.add(inp_embedded, noise_embed)
        # embedded = torch.cat((embedded, prev_ht), dim=2)
        # ht, fixed_hidden = self.lstm(embedded, hidden)
        # prob_ht = self.linear_ponder(ht)
        # # calulate N attn
        # attn = torch.bmm(ht, context.transpose(1, 2))
        # attn = F.softmax(attn.view(-1, self.grammar_len), dim=1).view(batch_size, -1, self.grammar_len)
        #
        # # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        # mix = torch.bmm(attn, context)
        #
        # # concat -> (batch, out_len, 2*dim)
        # combined = torch.cat((mix, ht), dim=2)
        # # output -> (batch, out_len, dim)
        # ht = F.tanh(self.linear_out(combined.view(-1, 2 * self.hidden_size))).view(
        #   batch_size, -1, self.hidden_size)
        # # gather the pondered hidden states and its probability
        # pondered_ht[0] = ht
        # pondered_prob_ht[0] = prob_ht
        # pondered_hidden[0] = fixed_hidden[0]
        # pondered_cell[0] = fixed_hidden[1]
        # vis_attn[0] = attn

        for c in range(0, self.n_k_factors):
          noise = Variable((c) * torch.ones(batch_size, 1).long()).cuda()
          noise_embed = self.ponder_noise(noise)  # [batch_size, 1, n_hidden]
          embedded = torch.add(inp_embedded, noise_embed)
          #embedded = torch.cat((embedded, prev_ht), dim=2)
          ht, hidden = self.lstm(embedded, hidden)
          prob_ht = self.linear_ponder(ht)
          # calulate N attn
          attn = torch.bmm(ht, context.transpose(1, 2))
          attn = F.softmax(attn.view(-1, self.grammar_len), dim=1).view(batch_size, -1, self.grammar_len)

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
        pondered_prob_hid = pondered_prob_ht.view(self.n_k_factors, 1, batch_size, 1)
        hid = torch.mul(pondered_hidden, pondered_prob_hid)
        hid = hid.sum(0).view(self.n_layers, batch_size, -1)
        cell = torch.mul(pondered_cell, pondered_prob_hid)
        cell = cell.sum(0).view(self.n_layers, batch_size, -1)
        hidden = (hid, cell)

      ### attention with dot mechanism Luong ###
      else:
        #embedded = torch.cat((embedded, prev_ht), dim=2)
        ht, hidden = self.lstm(embedded, hidden)  # [batch_size, 1, hidden_size]
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        # in_len is the number of characters in total k-factors
        # out_len is the number of step which is 1
        attn = torch.bmm(ht, context.transpose(1, 2))
        attn = F.softmax(attn.view(-1, self.grammar_len), dim=1).view(batch_size, -1, self.grammar_len)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, ht), dim=2)
        # output -> (batch, out_len, dim)
        ht = F.tanh(self.linear_out(combined.view(-1, 2 * self.hidden_size))).view(
          batch_size, -1, self.hidden_size)
        vis_attn[0] = attn

      output = self.output(ht.squeeze(1))

    return output, ht, hidden, vis_attn

  def init_hidden(self, batch_size):
    h0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
    c0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
    return h0, c0

  def init_prev_ht(self, batch_size, seq_len):
    ht = Variable(torch.zeros(batch_size, seq_len, self.hidden_size)).cuda()
    return ht