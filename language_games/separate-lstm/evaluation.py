import torch.nn.functional as F
import torch
from torch.autograd import Variable

def accuracy_test_data(dataset, decoder, inps_t, labels_t, attn=False, ponder=False):
  batch_size = 200
  it = len(inps_t) / batch_size
  acc_tot = 0
  acc_tot_seq = 0
  for i in range(int(it)):
    hidden = decoder.init_hidden(batch_size)
    decoder.zero_grad()
    teacher_forcing_ratio = 0.5
    start_index = i * batch_size
    inp, target = dataset.generate_batch(start_index, dataset.len_example, batch_size, inps_t, labels_t)
    if attn:
      i = 0
      output, ht, hidden, vis_attn = decoder(inp, hidden, batch_size)
      grammar_contexts = ht
      ht = decoder.init_prev_ht(batch_size, 1)
      # choose whether to use teacher forcing #
      # use_teacher_forcing = random.random() < teacher_forcing_ratio
      use_teacher_forcing = False
      dec_input = Variable(torch.ones(batch_size).long() * dataset.all_characters.index("SOS")).cuda()
      pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      for c in range(dataset.len_targets):
        if ponder:
          output, ht, hidden, vis_attn = decoder(target[:, c].unsqueeze(1), hidden, batch_size, attn=True,
                                                 context=grammar_contexts, prev_ht=ht, ponder=True)
        else:
          if use_teacher_forcing:
            output, ht, hidden, vis_attn = decoder(dec_input.unsqueeze(1), hidden, batch_size, attn=True,
                                                   context=grammar_contexts, prev_ht=ht, ponder=False)
            dec_input = target[:, c]
            pred_seq[c] = target[:, c]
            tgt_seq[c] = output.max(1)[1]
            accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
            acc_tot += accuracy.data[0]
          else:
            output, ht, hidden, vis_attn = decoder(dec_input.unsqueeze(1), hidden, batch_size, attn=True,
                                                   context=grammar_contexts, prev_ht=ht, ponder=False)
            # Get most likely word index (highest value) from output
            topv, topi = output.data.topk(1)
            top_pred = Variable(topi.squeeze(1)).cuda()
            # test = Variable(torch.ones(batch_size).long() * topi).cuda()
            dec_input = top_pred
            pred_seq[c] = target[:, c]
            tgt_seq[c] = output.max(1)[1]
            accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
            acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * 23
      # print(pred_seq, tgt_seq)
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
    else:
      output, ht, hidden, _ = decoder(inp, hidden, batch_size)
      # choose whether to use teacher forcing #
      # use_teacher_forcing = random.random() < teacher_forcing_ratio
      use_teacher_forcing = False
      dec_input = Variable(torch.ones(batch_size).long() * dataset.all_characters.index("SOS")).cuda()
      pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      for c in range(dataset.len_targets):
        if use_teacher_forcing:
          output, ht, hidden, _ = decoder(dec_input.unsqueeze(1), hidden, batch_size)
          dec_input = target[:, c]
          pred_seq[c] = target[:, c]
          tgt_seq[c] = output.max(1)[1]
          accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
          acc_tot += accuracy.data[0]
        else:
          output, ht, hidden, _ = decoder(dec_input.unsqueeze(1), hidden, batch_size)
          # Get most likely word index (highest value) from output
          topv, topi = output.data.topk(1)
          top_pred = Variable(topi.squeeze(1)).cuda()
          # test = Variable(torch.ones(batch_size).long() * topi).cuda()
          dec_input = top_pred
          pred_seq[c] = target[:, c]
          tgt_seq[c] = output.max(1)[1]
          accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
          acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * 23
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
  return acc_tot / (it * dataset.len_targets), acc_tot_seq / it


def accuracy_train_data(dataset, decoder, inps, targets, batch_size, attn=False, ponder=False):
  it = len(inps) / batch_size
  acc_tot = 0
  acc_tot_seq = 0
  for i in range(int(it)):
    hidden = decoder.init_hidden(batch_size)
    decoder.zero_grad()
    teacher_forcing_ratio = 0.5
    start_index = i * batch_size
    inp, target = dataset.generate_batch(start_index, dataset.len_example, batch_size, inps, targets)
    if attn:
      i = 0
      output, ht, hidden, vis_attn = decoder(inp, hidden, batch_size)
      grammar_contexts = ht
      ht = decoder.init_prev_ht(batch_size, 1)
      # choose whether to use teacher forcing #
      # use_teacher_forcing = random.random() < teacher_forcing_ratio
      use_teacher_forcing = False
      dec_input = Variable(torch.ones(batch_size).long() * dataset.all_characters.index("SOS")).cuda()
      pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      for c in range(dataset.len_targets):
        if ponder:
          output, ht, hidden, vis_attn = decoder(target[:, c].unsqueeze(1), hidden, batch_size, attn=True,
                                                 context=grammar_contexts, prev_ht=ht, ponder=True)
        else:
          if use_teacher_forcing:
            output, ht, hidden, vis_attn = decoder(dec_input.unsqueeze(1), hidden, batch_size, attn=True,
                                                   context=grammar_contexts, prev_ht=ht, ponder=False)
            dec_input = target[:, c]
            pred_seq[c] = target[:, c]
            tgt_seq[c] = output.max(1)[1]
            accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
            acc_tot += accuracy.data[0]
          else:
            output, ht, hidden, vis_attn = decoder(dec_input.unsqueeze(1), hidden, batch_size, attn=True,
                                                   context=grammar_contexts, prev_ht=ht, ponder=False)
            # Get most likely word index (highest value) from output
            topv, topi = output.data.topk(1)
            top_pred = Variable(topi.squeeze(1)).cuda()
            # test = Variable(torch.ones(batch_size).long() * topi).cuda()
            dec_input = top_pred
            # print(output)
            pred_seq[c] = target[:, c]
            tgt_seq[c] = output.max(1)[1]
            accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
            acc_tot += accuracy.data[0]
      # print(pred_seq, tgt_seq)
      truth = Variable(torch.ones(batch_size)).cuda() * 23
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
    else:
      output, ht, hidden, _ = decoder(inp, hidden, batch_size)
      # choose whether to use teacher forcing #
      # use_teacher_forcing = random.random() < teacher_forcing_ratio
      use_teacher_forcing = False
      dec_input = Variable(torch.ones(batch_size).long() * dataset.all_characters.index("SOS")).cuda()
      pred_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      tgt_seq = Variable(torch.zeros(dataset.len_targets, batch_size)).cuda()
      for c in range(dataset.len_targets):
        if use_teacher_forcing:
          output, ht, hidden, _ = decoder(dec_input.unsqueeze(1), hidden, batch_size)
          dec_input = target[:, c]
          pred_seq[c] = target[:, c]
          tgt_seq[c] = output.max(1)[1]
          accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
          acc_tot += accuracy.data[0]
        else:
          output, ht, hidden, _ = decoder(dec_input.unsqueeze(1), hidden, batch_size)
          # Get most likely word index (highest value) from output
          topv, topi = output.data.topk(1)
          top_pred = Variable(topi.squeeze(1)).cuda()
          # test = Variable(torch.ones(batch_size).long() * topi).cuda()
          dec_input = top_pred
          pred_seq[c] = target[:, c]
          tgt_seq[c] = output.max(1)[1]
          accuracy = (output.max(1)[1] == target[:, c]).float().sum().float() / batch_size
          acc_tot += accuracy.data[0]
      truth = Variable(torch.ones(batch_size)).cuda() * 23
      acc_tot_seq += ((pred_seq == tgt_seq).float().sum(dim=0) == truth).float().sum().data[0] / batch_size
  return acc_tot / (it * dataset.len_targets), acc_tot_seq / it