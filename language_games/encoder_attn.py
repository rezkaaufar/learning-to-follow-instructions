import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Encoder, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size

    self.embedding = nn.Embedding(input_size, hidden_size)
    self.attention_1 = nn.Parameter(torch.rand(1, hidden_size))
    self.attention_2 = nn.Parameter(torch.rand(1, hidden_size))
    self.attention_3 = nn.Parameter(torch.rand(1, hidden_size))

  def forward(self, inputs, batch_size, attn=False):
    # Note: we run this all at once (over the whole input sequence)
    pos_index = torch.LongTensor([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 17]).cuda()
    col_index = torch.LongTensor([8, 9, 14, 15]).cuda()
    com_index = torch.LongTensor([6, 16]).cuda()
    embedded = self.embedding(inputs)  # [batch_size, seq_len, hidden_size]if attn:
    if not attn:
      alpha_1 = torch.bmm(embedded, self.attention_1.unsqueeze(0).expand(batch_size, -1, -1).transpose(1,2))
      result_1 = torch.sum(alpha_1 * embedded, dim=1)
      alpha_2 = torch.bmm(embedded, self.attention_2.unsqueeze(0).expand(batch_size, -1, -1).transpose(1,2))
      result_2 = torch.sum(alpha_2 * embedded, dim=1)
      alpha_3 = torch.bmm(embedded, self.attention_3.unsqueeze(0).expand(batch_size, -1, -1).transpose(1,2))
      result_3 = torch.sum(alpha_3 * embedded, dim=1)
    else:
      result_1 = torch.sum(self.embedding.weight[(com_index)].mean(0) * embedded, dim=1)
      result_2 = torch.sum(self.embedding.weight[(col_index)].mean(0) * embedded, dim=1)
      result_3 = torch.sum(self.embedding.weight[(pos_index)].mean(0) * embedded, dim=1)
    return torch.cat([result_1.unsqueeze(1), result_2.unsqueeze(1), result_3.unsqueeze(1)], dim=1)