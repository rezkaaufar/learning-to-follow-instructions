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

  def forward(self, inputs, batch_size):
    # Note: we run this all at once (over the whole input sequence)
    embedded = self.embedding(inputs)  # [batch_size, seq_len, hidden_size]
    alpha_1 = torch.bmm(embedded, self.attention_1.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2))
    result_1 = torch.sum(alpha_1 * embedded, dim=1)
    alpha_2 = torch.bmm(embedded, self.attention_2.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2))
    result_2 = torch.sum(alpha_2 * embedded, dim=1)
    alpha_3 = torch.bmm(embedded, self.attention_3.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2))
    result_3 = torch.sum(alpha_3 * embedded, dim=1)
    return torch.cat([result_1.unsqueeze(1), result_2.unsqueeze(1), result_3.unsqueeze(1)], dim=1)