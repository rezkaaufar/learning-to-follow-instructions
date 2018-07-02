import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import data_loader
import RNN
import train
import evaluation
import math
import os
import itertools
import pandas as pd
from tensorboardX import SummaryWriter

## hyperparameters ##
n_epochs = 40
n_hidden = 128
n_layers = 2
lr = 1e-3
clip = 0.25
batch_size = 200
dropout = 0.4
grammar_len = 20
n_k_factors = 5
print_every = 200
n_categories = 2
len_example = 40 # length of the grammars and example
n_letters = 8
load = False

fseed = True

if fseed:
  torch.cuda.manual_seed_all(999)

dirs = os.path.dirname(os.path.abspath(__file__))

## helper function ##
def time_since(since):
  now = time.time()
  s = now - since
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

## main ##
# conf = [[32, 64, 128, 256],[1,2],[0, 0.2, 0.5]]
# config = list(itertools.product(*conf))
# config = []
# config.append((64, 1, 0.5))

# def run_train(config):
#   for j, elem in enumerate(config):
#     mn = [str(el) for el in elem]
#     model_name = "-".join(mn)
#     model_name = "LSTM" + which_data + "_" + model_name
#
#     # initialize model
#     # if not load:
#     attn = False
#     ponder = False
#     pool = False
#
#     rnn = RNN.RNN(dataset._n_letters, elem[0], dataset._n_categories, n_layers=elem[1],
#                   dropout_p=elem[2], grammar_len=grammar_len, n_k_factors=n_k_factors)
#     rnn.cuda()
#     # else:
#     #   rnn = torch.load('./models/lstm_5000example_20000c_attn_ponder_nvlex.tar').cuda()
#
#     rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#
#     start = time.time()
#     iters = 0
#     rnn.train(True)
#     losses, accs, accs_tr = [], [], []
#
#     writer = SummaryWriter()
#
#     for epoch in range(1, n_epochs + 1):
#       start_index = 0
#       steps = len(inps) / batch_size
#       for i in range(int(steps)):
#         start_index = i * batch_size
#         inp, target = dataset.generate_batch(start_index, batch_size, train=0)
#         loss, _, _ = train.train(inp, target, rnn, rnn_optimizer, criterion, batch_size, len_example, elem[0],
#                                  attn=attn, ponder=ponder, pool=pool)
#         writer.add_scalar('data/loss', loss, iters)
#         losses.append(loss)
#         if iters % print_every == 0:
#           acc = evaluation.accuracy_test_data(dataset, rnn, batch_size, len_example, elem[0], attn=attn, ponder=ponder,
#                                               pool=pool)
#           acc_v = evaluation.accuracy_valid_data(dataset, rnn, batch_size, len_example, elem[0], attn=attn, ponder=ponder,
#                                               pool=pool)
#           acc_tr = evaluation.accuracy_train_data(dataset, rnn, batch_size, len_example, elem[0], attn=attn, ponder=ponder,
#                                                   pool=pool)
#           accs.append(acc)
#           accs_tr.append(acc_tr)
#           writer.add_scalar('data/test_accuracy', acc, iters)
#           writer.add_scalar('data/valid_accuracy', acc_v, iters)
#           writer.add_scalar('data/train_accuracy', acc_tr, iters)
#           print("Loss {}, Test Accuracy {}, Valid Accuracy {}, Train Accuracy {}".format(loss, acc, acc_v, acc_tr))
#           # print("Loss {}".format(loss))
#
#         iters += 1
#
#     acc = evaluation.accuracy_test_data(dataset, rnn, batch_size, len_example, elem[0], attn=attn, ponder=ponder, pool=pool)
#     acc_v = evaluation.accuracy_valid_data(dataset, rnn, batch_size, len_example, elem[0], attn=attn, ponder=ponder, pool=pool)
#     acc_tr = evaluation.accuracy_train_data(dataset, rnn, batch_size, len_example, elem[0], attn=attn, ponder=ponder, pool=pool)
#     writer.add_scalar('data/test_accuracy', acc, iters)
#     writer.add_scalar('data/valid_accuracy', acc_v, iters)
#     writer.add_scalar('data/train_accuracy', acc_tr, iters)
#     writer.export_scalars_to_json("json/lstm/" + model_name + ".json")
#     writer.close()
#
#     with open('./models/lstm/'+ model_name + '.tar', 'wb') as ckpt:
#       torch.save(rnn, ckpt)

# run_train(config)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--hidden_size', type=int, choices=[32,64,128,256])
ap.add_argument('--dropout_rate', type=float, choices=[0.0, 0.2, 0.5])
ap.add_argument('--layers_lstm', type=int, choices=[1, 2])
ap.add_argument('--attention', type=int, choices=[1, 2])
ap.add_argument('--pondering', type=int, choices=[1, 2])
ap.add_argument('--data')
ap.add_argument('--ver')
args = ap.parse_args()

# initialize dataset
#which_data = "1000"
which_data = args.data
data_version = args.ver
datver_1 = ""
datver_2 = ""
if args.ver == "non":
  datver_1 = data_version + "-"
  datver_2 = "_" + data_version

inps, labels = data_loader.read_data(dirs + "/dataset/" + datver_1 + "overlap/subreg_train_" + which_data + datver_2 + ".txt")
# inps_v, labels_v = data_loader.read_data(dirs + "/dataset/non-overlap/subreg_valid_" + which_data + "_non.txt")
inps_v, labels_v = data_loader.read_data(dirs + "/dataset/" + datver_1 + "overlap/subreg_valid_" + which_data + datver_2 + ".txt")
inps_t, labels_t = data_loader.read_data(dirs + "/dataset/" + datver_1 + "overlap/subreg_test_" + which_data + datver_2 + ".txt")
dataset = data_loader.Dataset(inps, labels, inps_v, labels_v, inps_t, labels)
dataset.randomize_data()
dataset.randomize_data_ev()
dataset.randomize_data_odd()

print(dirs + "/dataset/" + datver_1 + "overlap/subreg_valid_" + which_data + datver_2 + ".txt")

attn = False
if args.attention == 2:
  attn = True
ponder = False
if args.pondering == 2:
  ponder = True
model_path = ""
if attn and ponder:
  model_path = "lstm+attn+act"
elif attn:
  model_path = "lstm+attn"
else:
  model_path = "lstm"
model_name = "-".join(["{}_{}".format(k, getattr(args, k)) for k in vars(args) if getattr(args, k) is not None])
model_name = "LSTM_" + which_data + "_" + model_name
print(model_name)
print(model_path)

pool = False

rnn = RNN.RNN(dataset._n_letters, args.hidden_size, dataset._n_categories, n_layers=args.layers_lstm,
              dropout_p=args.dropout_rate, grammar_len=grammar_len, n_k_factors=n_k_factors)
rnn.cuda()

rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


start = time.time()
iters = 0
rnn.train(True)
losses, accs, accs_tr = [], [], []

writer = SummaryWriter()
df = pd.DataFrame(columns=["data/loss", 'data/test_accuracy', 'data/test_reject_accuracy', 'data/test_accept_accuracy'
'data/train_accuracy', 'data/valid_accuracy', 'data/valid_reject_accuracy', 'data/valid_accept_accuracy', 'iters'])
df = df.set_index('iters')

cur_best = 0
for epoch in range(1, n_epochs + 1):
  start_index = 0
  steps = len(inps) / batch_size
  for i in range(int(steps)):
    start_index = i * batch_size
    inp, target = dataset.generate_batch(start_index, batch_size, train=0)
    loss, _, _ = train.train(inp, target, rnn, rnn_optimizer, criterion, batch_size, len_example, args.hidden_size,
                             attn=attn, ponder=ponder, pool=pool)
    writer.add_scalar('data/loss', loss, iters)
    losses.append(loss)
    if iters % print_every == 0:
      acc = evaluation.accuracy_test_data(dataset, rnn, batch_size, len_example, args.hidden_size, attn=attn, ponder=ponder,
                                          pool=pool)
      acc_te = evaluation.accuracy_data(dataset, 2, False, rnn, batch_size, len_example, args.hidden_size, attn=attn,
                                        ponder=ponder,
                                        pool=pool)
      acc_to = evaluation.accuracy_data(dataset, 2, True, rnn, batch_size, len_example, args.hidden_size, attn=attn,
                                        ponder=ponder,
                                        pool=pool)
      acc_v = evaluation.accuracy_valid_data(dataset, rnn, batch_size, len_example, args.hidden_size, attn=attn,
                                             ponder=ponder,
                                             pool=pool)
      acc_ve = evaluation.accuracy_data(dataset, 1, False, rnn, batch_size, len_example, args.hidden_size, attn=attn,
                                        ponder=ponder,
                                        pool=pool)
      acc_vo = evaluation.accuracy_data(dataset, 1, True, rnn, batch_size, len_example, args.hidden_size, attn=attn,
                                        ponder=ponder,
                                        pool=pool)
      acc_tr = evaluation.accuracy_train_data(dataset, rnn, batch_size, len_example, args.hidden_size, attn=attn, ponder=ponder,
                                              pool=pool)
      writer.add_scalar('data/test_accuracy', acc, iters)
      writer.add_scalar('data/test_reject_accuracy', acc_to, iters)
      writer.add_scalar('data/test_accept_accuracy', acc_te, iters)
      writer.add_scalar('data/valid_accuracy', acc_v, iters)
      writer.add_scalar('data/valid_reject_accuracy', acc_vo, iters)
      writer.add_scalar('data/valid_accept_accuracy', acc_ve, iters)
      writer.add_scalar('data/train_accuracy', acc_tr, iters)
      df.loc[iters, 'data/test_accuracy'] = acc
      df.loc[iters, 'data/test_reject_accuracy'] = acc_to
      df.loc[iters, 'data/test_accept_accuracy'] = acc_te
      df.loc['data/train_accuracy'] = acc_tr
      df.loc['data/valid_accuracy'] = acc_v
      df.loc[iters, 'data/valid_reject_accuracy'] = acc_vo
      df.loc[iters, 'data/valid_accept_accuracy'] = acc_ve
      if acc_v > cur_best:
        print("Writing models at epoch {}".format(epoch))
        with open(dirs + '/models/' + model_path + '/' + model_name + '.tar', 'wb') as ckpt:
          torch.save(rnn, ckpt)
      print("Loss {}, Test Accuracy {}, Train Accuracy {}, Validation Accuracy {}".format(loss, acc, acc_tr, acc_v))
      print("Val Rej {}, Val Acc {}, Test Rej {}, Test Acc {}".format(acc_vo, acc_ve, acc_to, acc_te))

    iters += 1

acc = evaluation.accuracy_test_data(dataset, rnn, batch_size, len_example, args.hidden_size, attn=attn, ponder=ponder, pool=pool)
acc_te = evaluation.accuracy_data(dataset, 2, False, rnn, batch_size, len_example, args.hidden_size, attn=attn, ponder=ponder,
                                       pool=pool)
acc_to = evaluation.accuracy_data(dataset, 2, True, rnn, batch_size, len_example, args.hidden_size, attn=attn, ponder=ponder,
                                       pool=pool)
acc_v = evaluation.accuracy_valid_data(dataset, rnn, batch_size, len_example, args.hidden_size, attn=attn, ponder=ponder,
                                       pool=pool)
acc_ve = evaluation.accuracy_data(dataset, 1, False, rnn, batch_size, len_example, args.hidden_size, attn=attn, ponder=ponder,
                                       pool=pool)
acc_vo = evaluation.accuracy_data(dataset, 1, True, rnn, batch_size, len_example, args.hidden_size, attn=attn, ponder=ponder,
                                       pool=pool)
acc_tr = evaluation.accuracy_train_data(dataset, rnn, batch_size, len_example, args.hidden_size, attn=attn, ponder=ponder,
                                        pool=pool)
writer.add_scalar('data/test_accuracy', acc, iters)
writer.add_scalar('data/test_reject_accuracy', acc_to, iters)
writer.add_scalar('data/test_accept_accuracy', acc_te, iters)
writer.add_scalar('data/valid_accuracy', acc_v, iters)
writer.add_scalar('data/valid_reject_accuracy', acc_vo, iters)
writer.add_scalar('data/valid_accept_accuracy', acc_ve, iters)
writer.add_scalar('data/train_accuracy', acc_tr, iters)
df.loc[iters, 'data/test_accuracy'] = acc
df.loc[iters, 'data/test_reject_accuracy'] = acc_to
df.loc[iters, 'data/test_accept_accuracy'] = acc_te
df.loc['data/train_accuracy'] = acc_tr
df.loc['data/valid_accuracy'] = acc_v
df.loc[iters, 'data/valid_reject_accuracy'] = acc_vo
df.loc[iters, 'data/valid_accept_accuracy'] = acc_ve
writer.export_scalars_to_json(dirs + "/json/" + model_path + "/" + model_name + ".json")
df.to_json(dirs + "/json/" + model_path + "/" + model_name + "_pd.json")
writer.close()

# with open(dirs + '/models/' + model_path + '/' + model_name + '.tar', 'wb') as ckpt:
#   torch.save(rnn, ckpt)