import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import Decoder
import Encoder
import data_loader
import train
import evaluation
import math
import itertools
import os
import pandas as pd
from tensorboardX import SummaryWriter

## hyperparameters ##
# hyperparameters
n_epochs = 100
n_hidden = 256 # (32, 64, 128, 256)
n_layers = 2 # (1, 2)
layers_conv = 5
lr = 1e-3
clip = 5.0
batch_size = 200
dropout = 0.5
ponder_step = 5
concat = False
attn = True
dot_str = ""
bi = False
bi_str = ""
cloud = False
cloud_str = ""

if cloud:
  cloud_str = "/home/rezkaauf/language_games/"
if attn:
  if concat:
    dot_str = "_concat"
  else:
    dot_str = "_dot"
else:
  dot_str = "_noattn"

if bi:
  bi_str = "_decbi"

print_every = 200
load = False

fseed = True

if fseed:
  torch.cuda.manual_seed_all(999)

dirs = os.path.dirname(os.path.abspath(__file__))
dirs += "/.."

## helper function ##
def time_since(since):
  now = time.time()
  s = now - since
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

## main ##
# initialize dataset
which_data = "utter_blocks"
inps, instrs, targets = data_loader.read_data(dirs + "/dataset/lang_games_data_artificial_train_nvl_"
                                              + which_data + "_24000_3.txt")
inps_v, instrs_v, targets_v = data_loader.read_data(dirs + "/dataset/lang_games_data_artificial_valid_nvl_"
                                                    + which_data + "_24000_3.txt")
inps_t, instrs_t, targets_t = data_loader.read_data(dirs + "/dataset/lang_games_data_artificial_test_nvl_"
                                                    + which_data + "_24000_3.txt")
dataset = data_loader.Dataset(inps, instrs, targets, inps_v, instrs_v, targets_v, inps_t, instrs_t, targets_t)
dataset.randomize_data()

# initialize model
# conf = [[256],[2],[0.5]]
# config = list(itertools.product(*conf))
config = []
config.append((64, 1, 0.5))
# config.append((64, 2, 0.5))
# config.append((128, 2, 0))
# config.append((128, 2, 0.2))
# config.append((128, 2, 0.5))
# config.append((256, 2, 0.5))
#print(config)
#config = [[128,2,0.5]]

def run_train(config):
  for j, elem in enumerate(config):
    model_name = "Trans2Conv_50000_nvl_" + which_data + "_hid" + str(elem[0]) + \
                 "_layer" + str(elem[1]) + "_drop" + str(elem[2]) + dot_str + bi_str + "_new"
    batch_size = 200
    if load:
      decoder = torch.load("models/" + model_name + ".tar")
    else:
      decoder = Decoder.ConvDecoder(dataset.n_letters, elem[0], elem[0], dataset.n_letters, dataset.len_example,
                                    kernel_size=3, n_layers=layers_conv, dropout_p=elem[2], example_len=dataset.len_instr)
    encoder = Encoder.EncoderWord(dataset.n_words, elem[0], n_layers=elem[1])
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    decoder.cuda()
    encoder.cuda()
    start = time.time()
    iters = 0
    losses, accs, accs_tr = [], [], []

    writer = SummaryWriter()
    df = pd.DataFrame(columns=["data/loss", 'data/test_accuracy',
    'data/train_accuracy', 'data/test_seq_accuracy', 'data/train_seq_accuracy',
    'data/val_accuracy', 'data/val_seq_accuracy', 'iters'])
    df = df.set_index('iters')

    for epoch in range(1, n_epochs + 1):
      decoder.train(True)
      encoder.train(True)
      steps = len(inps) / batch_size
      for i in range(int(steps)):
        start_index = i * batch_size
        inp, instr, target = dataset.generate_batch(start_index, batch_size, inps, instrs, targets)
        loss = train.train(dataset, encoder, decoder, enc_optimizer, optimizer, criterion, dataset.len_targets, batch_size,
                           inp, instr, target, attn=attn)
        writer.add_scalar('data/loss', loss, iters)
        df.loc[iters,'data/loss'] = loss
        losses.append(loss)
        iters += 1
      acc, acc_seq = evaluation.accuracy_test_data(dataset, encoder, decoder, inps_t, instrs_t, targets_t,
                                                   batch_size, attn=attn)
      acc_val, acc_val_seq = evaluation.accuracy_test_data(dataset, encoder, decoder, inps_v, instrs_v, targets_v,
                                                           batch_size, attn=attn)
      acc_tr, acc_tr_seq = evaluation.accuracy_train_data(dataset, encoder, decoder, inps, instrs, targets,
                                                          batch_size, attn=attn)
      writer.add_scalar('data/test_accuracy', acc, iters)
      writer.add_scalar('data/train_accuracy', acc_tr, iters)
      writer.add_scalar('data/test_seq_accuracy', acc_seq, iters)
      writer.add_scalar('data/train_seq_accuracy', acc_tr_seq, iters)
      writer.add_scalar('data/val_accuracy', acc_val, iters)
      writer.add_scalar('data/val_seq_accuracy', acc_val_seq, iters)
      df.loc[iters,'data/test_accuracy'] = acc
      df.loc['data/train_accuracy'] = acc_tr
      df.loc['data/test_seq_accuracy'] = acc_seq
      df.loc['data/train_seq_accuracy'] = acc_tr_seq
      df.loc['data/val_accuracy'] = acc_val
      df.loc['data/val_seq_accuracy'] = acc_val_seq
      if acc_val_seq > cur_best:
        cur_best = acc_val_seq
        print("Writing models at epoch {}".format(epoch))
        with open(dirs + "/models/conv2seq/" + "Encoder_" + model_name + ".tar", 'wb') as ckpt:
          torch.save(encoder, ckpt)
        with open(dirs + "/models/conv2seq/" + "Decoder_" + model_name + ".tar", 'wb') as ckpt:
          torch.save(decoder, ckpt)
        torch.save(decoder.state_dict(),
                   dirs + '/models/conv2seq/Params_Decoder_' + model_name + '.tar')
        torch.save(encoder.state_dict(),
                   dirs + '/models/conv2seq/Params_Encoder_' + model_name + '.tar')
      #accs.append(acc_seq)
      #accs_tr.append(acc_tr_seq)
      print("Config {}, Loss {}, Test Accuracy {}, Train Accuracy {}, Val Accuracy {}, "
            "Test Seq Accuracy {}, Train Seq Accuracy {}, Val Seq Accuracy {}"
        .format(str(j), loss, acc, acc_tr, acc_val, acc_seq, acc_tr_seq, acc_val_seq))

    writer.export_scalars_to_json(dirs + "/json/"+ model_name +".json")
    writer.close()
    df.to_json(dirs + '/json/'+ model_name + '.pd.json')

#run_train(config)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--hidden_size', type=int, choices=[32,64,128,256])
ap.add_argument('--dropout_rate', type=float, choices=[0.0, 0.2, 0.5])
ap.add_argument('--layers_lstm', type=int, choices=[1, 2])

args = ap.parse_args()

model_name = "-".join(["{}_{}".format(k, getattr(args, k)) for k in vars(args) if getattr(args, k) is not None])
model_name = "Seq2Seq_24000_3_nvl_" + which_data + "_" + model_name

t_start = time.time()

decoder = Decoder.Decoder(dataset.n_letters, args.hidden_size, dataset.n_letters, args.layers_lstm, dropout_p=args.dropout_rate,
                          example_len=dataset.len_instr, concat=False)
encoder = Encoder.Encoder(dataset.n_words, args.hidden_size, args.layers_lstm, dropout_p=args.dropout_rate)
enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.NLLLoss()
decoder.cuda()
encoder.cuda()
start = time.time()
iters = 0
losses, accs, accs_tr = [], [], []

writer = SummaryWriter()
df = pd.DataFrame(columns=["data/loss", 'data/test_accuracy',
'data/train_accuracy', 'data/test_seq_accuracy', 'data/train_seq_accuracy',
'data/val_accuracy', 'data/val_seq_accuracy', 'iters'])
df = df.set_index('iters')

cur_best = 0
for epoch in range(1, n_epochs + 1):
  decoder.train(True)
  encoder.train(True)
  steps = len(inps) / batch_size
  for i in range(int(steps)):
    start_index = i * batch_size
    inp, instr, target = dataset.generate_batch(start_index, batch_size, inps, instrs, targets)
    loss = train.train(dataset, encoder, decoder, enc_optimizer, optimizer, criterion, dataset.len_targets, batch_size,
                       inp, instr, target, attn=attn)
    writer.add_scalar('data/loss', loss, iters)
    df.loc[iters,'data/loss'] = loss
    losses.append(loss)
    iters += 1
  acc, acc_seq = evaluation.accuracy_test_data(dataset, encoder, decoder, inps_t, instrs_t, targets_t,
                                               batch_size, attn=attn)
  acc_val, acc_val_seq = evaluation.accuracy_test_data(dataset, encoder, decoder, inps_v, instrs_v, targets_v,
                                                       batch_size, attn=attn)
  acc_tr, acc_tr_seq = evaluation.accuracy_train_data(dataset, encoder, decoder, inps, instrs, targets,
                                                      batch_size, attn=attn)
  writer.add_scalar('data/test_accuracy', acc, iters)
  writer.add_scalar('data/train_accuracy', acc_tr, iters)
  writer.add_scalar('data/test_seq_accuracy', acc_seq, iters)
  writer.add_scalar('data/train_seq_accuracy', acc_tr_seq, iters)
  writer.add_scalar('data/val_accuracy', acc_val, iters)
  writer.add_scalar('data/val_seq_accuracy', acc_val_seq, iters)
  df.loc[iters,'data/test_accuracy'] = acc
  df.loc['data/train_accuracy'] = acc_tr
  df.loc['data/test_seq_accuracy'] = acc_seq
  df.loc['data/train_seq_accuracy'] = acc_tr_seq
  df.loc['data/val_accuracy'] = acc_val
  df.loc['data/val_seq_accuracy'] = acc_val_seq
  if acc_val_seq > cur_best:
    cur_best = acc_val_seq
    print("Writing models at epoch {}".format(epoch))
    with open(dirs + "/models/seq2seq/" + "Encoder_" + model_name + ".tar", 'wb') as ckpt:
      torch.save(encoder, ckpt)
    with open(dirs + "/models/seq2seq/" + "Decoder_" + model_name + ".tar", 'wb') as ckpt:
      torch.save(decoder, ckpt)
    torch.save(decoder.state_dict(),
               dirs + '/models/seq2seq/Params_Decoder_' + model_name + '.tar')
    torch.save(encoder.state_dict(),
               dirs + '/models/seq2seq/Params_Encoder_' + model_name + '.tar')
  # accs.append(acc_seq)
  # accs_tr.append(acc_tr_seq)
  print("Loss {}, Test Accuracy {}, Train Accuracy {}, Val Accuracy {}, "
        "Test Seq Accuracy {}, Train Seq Accuracy {}, Val Seq Accuracy {}"
        .format(loss, acc, acc_tr, acc_val, acc_seq, acc_tr_seq, acc_val_seq))

writer.export_scalars_to_json(dirs + "/json/seq2seq/" + model_name + ".json")
writer.close()
df.to_json(dirs + '/json/seq2seq/'+ model_name + '.pd.json')
