import torch.nn.functional as F
import torch
from torch.autograd import Variable

def generate_position_ids_word(batch_size, len_instr):
  pos_tensor = torch.zeros(batch_size, len_instr).long()
  for i in range(batch_size):
    pos_tensor[i] = torch.LongTensor(range(0, len_instr))
  return Variable(pos_tensor).cuda()

def accuracy_test_data(dataset, encoder, decoder, inps_t, instrs_t, targets_t, batch_size, attn=False):
  it = len(inps_t) / batch_size
  acc_tot = 0
  acc_tot_seq = 0
  for i in range(int(it)):
    decoder.eval()
    encoder.eval()
    start_index = i * batch_size
    inp, instr, target = dataset.generate_batch(start_index, batch_size, inps_t, instrs_t, targets_t)
    pos_id_words = generate_position_ids_word(batch_size, dataset.len_instr)
    encoder_ht = encoder(instr, pos_id_words, batch_size)
    context = encoder_ht
    hidden = decoder.init_hidden(batch_size)
    if attn:
      pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      for c in range(dataset.len_targets):
        output, ht, hidden, vis_attn = decoder(inp[:, c].unsqueeze(1), hidden, batch_size, attn=True,
                                               context=context)
        tgt_seq[c] = target[:, c]
        pred_seq[c] = output.max(1)[1]
        accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
        acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * 23
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
      # print((pred_seq == tgt_seq).float().sum(dim=0))
    else:
      pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      for c in range(dataset.len_targets):
        output, ht, hidden, vis_attn = decoder(inp[:, c].unsqueeze(1), hidden, batch_size)
        tgt_seq[c] = target[:, c]
        pred_seq[c] = output.max(1)[1]
        accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
        acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * 23
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
      # print((pred_seq == tgt_seq).float().sum(dim=0))
  return acc_tot / (it * dataset.len_targets), acc_tot_seq / it

def accuracy_train_data(dataset, encoder, decoder, inps, instrs, targets, batch_size, attn=False):
  it = len(inps) / batch_size
  acc_tot = 0
  acc_tot_seq = 0
  for i in range(int(it)):
    decoder.eval()
    encoder.eval()
    start_index = i * batch_size
    inp, instr, target = dataset.generate_batch(start_index, batch_size, inps, instrs, targets)
    pos_id_words = generate_position_ids_word(batch_size, dataset.len_instr)
    encoder_ht = encoder(instr, pos_id_words, batch_size)
    context = encoder_ht
    hidden = decoder.init_hidden(batch_size)
    if attn:
      pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      for c in range(dataset.len_targets):
        output, ht, hidden, vis_attn = decoder(inp[:, c].unsqueeze(1), hidden, batch_size, attn=True,
                                               context=context)
        tgt_seq[c] = target[:, c]
        pred_seq[c] = output.max(1)[1]
        accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
        acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * 23
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
      # print((pred_seq == tgt_seq).float().sum(dim=0))
    else:
      pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      for c in range(dataset.len_targets):
        output, ht, hidden, vis_attn = decoder(inp[:, c].unsqueeze(1), hidden, batch_size)
        tgt_seq[c] = target[:, c]
        pred_seq[c] = output.max(1)[1]
        accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
        acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * 23
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
      # print((pred_seq == tgt_seq).float().sum(dim=0))
  return acc_tot / (it * dataset.len_targets), acc_tot_seq / it