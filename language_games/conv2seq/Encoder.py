import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvEncoder(nn.Module):
  def __init__(self, input_size, hidden_size, num_channels, output_size, max_len, kernel_size=3, n_layers=2,
               dropout_p=0.2):
    super(ConvEncoder, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.input_dropout = nn.Dropout(p=dropout_p)
    self.embed = nn.Embedding(input_size, hidden_size)
    self.position_embedding = nn.Embedding(max_len, hidden_size)
    self.output = nn.Linear(hidden_size, output_size)
    self.len_instr = max_len

    self.conv = nn.ModuleList([nn.Conv1d(num_channels, num_channels, kernel_size,
                                         padding=kernel_size // 2) for _ in range(n_layers)])

  def forward(self, inputs, position_inp, batch_size):
    # Retrieving position and word embeddings
    position_embedding = self.position_embedding(position_inp)
    word_embedding = self.embed(inputs)

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
      cnn = F.tanh(layer(cnn) + cnn)
    return cnn.transpose(1, 2)