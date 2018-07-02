import torch.nn.functional as F
import torch
from torch.autograd import Variable

def accuracy_test_data(dataset, rnn, batch_size, len_example, n_hidden, attn=False, ponder=False, pool=False):
  it = len(dataset._inps_t) / batch_size
  acc_tot = 0
  rnn.eval()
  for i in range(int(it)):
    # pool
    pool_hidden = Variable(torch.zeros(len_example - 20, batch_size, 1
                                           , n_hidden)).cuda()
    hidden = rnn.init_hidden(batch_size)
    start_index = i * batch_size
    inp_t, lab_t = dataset.generate_batch(start_index, batch_size, train=2)

    if attn:
      output, ht, hidden, vis_attn= rnn(inp_t[:, :20], hidden, batch_size)
      grammar_contexts = ht
      ht = rnn.init_prev_ht(batch_size, 1)
      for c in range(len_example - 20):
        # print(ht)
        if ponder:
          output, ht, hidden, vis_attn = rnn(inp_t[:, 20 + c].unsqueeze(1), hidden, batch_size, attn=True,
                                   context=grammar_contexts, prev_ht=ht, ponder=True)
        else:
          output, ht, hidden, vis_attn = rnn(inp_t[:, 20 + c].unsqueeze(1), hidden, batch_size, attn=True,
                                   context=grammar_contexts, prev_ht=ht, ponder=False)
        pool_hidden[c] = ht
      # pool
      if pool:
          logits, _ = torch.max(pool_hidden, 0)
          cls = rnn.output(logits.squeeze(1))
      else:
          cls = output
      cls = F.softmax(cls, dim=1)
    else:
      output, ht, hidden, vis_attn = rnn(inp_t, hidden, batch_size)
      cls = output[:, -1, :].contiguous()
      cls = F.softmax(cls, dim=1)

    max_idx = cls.max(1)[1]
    accuracy = (lab_t.squeeze() == max_idx).long().sum().float() / batch_size
    acc_tot += accuracy.data[0]  # , max_idx.sum().data[0]
  return acc_tot / it

def accuracy_valid_data(dataset, rnn, batch_size, len_example, n_hidden, attn=False, ponder=False, pool=False):
  it = len(dataset._inps_v) / batch_size
  acc_tot = 0
  rnn.eval()
  for i in range(int(it)):
    # pool
    pool_hidden = Variable(torch.zeros(len_example - 20, batch_size, 1
                                           , n_hidden)).cuda()
    hidden = rnn.init_hidden(batch_size)
    start_index = i * batch_size
    inp_t, lab_t = dataset.generate_batch(start_index, batch_size, train=1)

    if attn:
      output, ht, hidden, vis_attn= rnn(inp_t[:, :20], hidden, batch_size)
      grammar_contexts = ht
      ht = rnn.init_prev_ht(batch_size, 1)
      for c in range(len_example - 20):
        # print(ht)
        if ponder:
          output, ht, hidden, vis_attn = rnn(inp_t[:, 20 + c].unsqueeze(1), hidden, batch_size, attn=True,
                                   context=grammar_contexts, prev_ht=ht, ponder=True)
        else:
          output, ht, hidden, vis_attn = rnn(inp_t[:, 20 + c].unsqueeze(1), hidden, batch_size, attn=True,
                                   context=grammar_contexts, prev_ht=ht, ponder=False)
        pool_hidden[c] = ht
      # pool
      if pool:
          logits, _ = torch.max(pool_hidden, 0)
          cls = rnn.output(logits.squeeze(1))
      else:
          cls = output
      cls = F.softmax(cls, dim=1)
    else:
      output, ht, hidden, vis_attn = rnn(inp_t, hidden, batch_size)
      cls = output[:, -1, :].contiguous()
      cls = F.softmax(cls, dim=1)

    max_idx = cls.max(1)[1]
    accuracy = (lab_t.squeeze() == max_idx).long().sum().float() / batch_size
    acc_tot += accuracy.data[0]  # , max_idx.sum().data[0]
  return acc_tot / it

def accuracy_train_data(dataset, rnn, batch_size, len_example, n_hidden, attn=False, ponder=False, pool=False):
  it = len(dataset._inps) / batch_size
  acc_tot = 0
  rnn.eval()
  for i in range(int(it)):
    # pool
    pool_hidden = Variable(torch.zeros(len_example - 20, batch_size, 1
                                           , n_hidden)).cuda()
    hidden = rnn.init_hidden(batch_size)
    start_index = i * batch_size
    inp_t, lab_t = dataset.generate_batch(start_index, batch_size, train=0)

    if attn:
      output, ht, hidden, vis_attn = rnn(inp_t[:, :20], hidden, batch_size)
      grammar_contexts = ht
      ht = rnn.init_prev_ht(batch_size, 1)
      for c in range(len_example - 20):
        if ponder:
          output, ht, hidden, vis_attn = rnn(inp_t[:, 20 + c].unsqueeze(1), hidden, batch_size, attn=True,
                                   context=grammar_contexts, prev_ht=ht, ponder=True)
        else:
          output, ht, hidden, vis_attn = rnn(inp_t[:, 20 + c].unsqueeze(1), hidden, batch_size, attn=True,
                                   context=grammar_contexts, prev_ht=ht, ponder=False)
        pool_hidden[c] = ht
      # pool
      if pool:
        logits, _ = torch.max(pool_hidden, 0)
        cls = rnn.output(logits.squeeze(1))
      else:
        cls = output
      cls = F.softmax(cls, dim=1)
    else:
      output, ht, hidden, vis_attn= rnn(inp_t, hidden, batch_size)
      cls = output[:, -1, :].contiguous()
      cls = F.softmax(cls, dim=1)

    max_idx = cls.max(1)[1]
    accuracy = (lab_t.squeeze() == max_idx).long().sum().float() / batch_size
    acc_tot += accuracy.data[0]  # , max_idx.sum().data[0]
  return acc_tot / it

def accuracy_data(dataset, train, odd, rnn, batch_size, len_example, n_hidden, attn=False, ponder=False, pool=False):
  it = 0
  if train == 1 and odd:
    it = len(dataset._inps_vo) / batch_size
  elif train == 1 and not odd:
    it = len(dataset._inps_ve) / batch_size
  elif train == 2 and odd:
    it = len(dataset._inps_to) / batch_size
  elif train == 2 and not odd:
    it = len(dataset._inps_te) / batch_size
  acc_tot = 0
  rnn.eval()
  for i in range(int(it)):
    # pool
    pool_hidden = Variable(torch.zeros(len_example - 20, batch_size, 1
                                           , n_hidden)).cuda()
    hidden = rnn.init_hidden(batch_size)
    start_index = i * batch_size
    if odd:
      inp_t, lab_t = dataset.generate_batch_odd(start_index, batch_size, train=train)
    else:
      inp_t, lab_t = dataset.generate_batch_even(start_index, batch_size, train=train)

    if attn:
      output, ht, hidden, vis_attn = rnn(inp_t[:, :20], hidden, batch_size)
      grammar_contexts = ht
      ht = rnn.init_prev_ht(batch_size, 1)
      for c in range(len_example - 20):
        if ponder:
          output, ht, hidden, vis_attn = rnn(inp_t[:, 20 + c].unsqueeze(1), hidden, batch_size, attn=True,
                                   context=grammar_contexts, prev_ht=ht, ponder=True)
        else:
          output, ht, hidden, vis_attn = rnn(inp_t[:, 20 + c].unsqueeze(1), hidden, batch_size, attn=True,
                                   context=grammar_contexts, prev_ht=ht, ponder=False)
        pool_hidden[c] = ht
      # pool
      if pool:
        logits, _ = torch.max(pool_hidden, 0)
        cls = rnn.output(logits.squeeze(1))
      else:
        cls = output
      cls = F.softmax(cls, dim=1)
    else:
      output, ht, hidden, vis_attn= rnn(inp_t, hidden, batch_size)
      cls = output[:, -1, :].contiguous()
      cls = F.softmax(cls, dim=1)

    max_idx = cls.max(1)[1]
    accuracy = (lab_t.squeeze() == max_idx).long().sum().float() / batch_size
    acc_tot += accuracy.data[0]  # , max_idx.sum().data[0]
  return acc_tot / it