import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvDecoder(nn.Module):
  def __init__(self, input_size, hidden_size, num_channels, output_size, max_len, kernel_size=3, n_layers=2,
               dropout_p=0.2, example_len=15):
    super(ConvDecoder, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.input_dropout = nn.Dropout(p=dropout_p)
    self.embed = nn.Embedding(input_size, hidden_size)
    self.position_embedding = nn.Embedding(max_len, hidden_size)
    self.output = nn.Linear(hidden_size, output_size)
    self.len_instr = example_len
    self.len_block = max_len

    self.conv = nn.ModuleList([nn.Conv1d(num_channels, num_channels, kernel_size,
                                         padding=kernel_size // 2) for _ in range(n_layers)])
    # for context-attended output
    self.linear_out = nn.Linear(hidden_size * 2, hidden_size)

  def forward(self, inputs, position_inp, batch_size, attn=False, context=None):
    # Retrieving position and word embeddings
    position_embedding = self.position_embedding(position_inp)
    word_embedding = self.embed(inputs)
    # for visualization #
    vis_attn = Variable(torch.zeros(self.len_block, batch_size, 1, self.len_instr)).cuda()

    # Applying dropout to the sum of position + word embeddings
    embedded = self.input_dropout(position_embedding + word_embedding)  # [batch_size, seq_len, embed_size]
    # Transform the input to be compatible for Conv1d as follows
    # Num Batches * Length * Channel ==> Num Batches * Channel * Length
    embedded = embedded.transpose(1, 2)

    # Successive application of convolution layers followed by residual connection
    # and non-linearity
    cnn = embedded
    for i, layer in enumerate(self.conv):
      # layer(cnn) is the convolution operation on the input cnn after which
      # we add the original input creating a residual connection
      # if attn:
      # Applying attention to the hidden states produced by the deep convolution
      ht = cnn.transpose(1, 2)  # batch_size, seq_len, hidden_size
      attn = torch.bmm(ht, context.transpose(1, 2))
      # print(attn.size())
      attn = F.softmax(attn.view(-1, self.len_instr), dim=1).view(batch_size, -1, self.len_instr)

      # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
      mix = torch.bmm(attn, context)

      # concat -> (batch, out_len, 2*dim)
      combined = torch.cat((mix, ht), dim=2)
      # output -> (batch, out_len, dim)
      ht = F.tanh(self.linear_out(combined.view(-1, 2 * self.hidden_size))).view(
        batch_size, -1, self.hidden_size)
      # print(ht.size())
      vis_attn = attn
      cnn = ht.transpose(1, 2)
      cnn = F.tanh(layer(cnn) + cnn)
    out_ht = cnn.transpose(1, 2)
    # print(out_ht.size())
    output = F.log_softmax(self.output(out_ht), dim=2)
    return output, vis_attn