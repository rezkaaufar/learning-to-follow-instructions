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
from tensorboardX import SummaryWriter

## hyperparameters ##
# hyperparameters
n_epochs = 100
n_hidden = 128 # (32, 64, 128, 256)
n_layers = 2 # (1, 2)
lr = 1e-3
clip = 5.0
batch_size = 200
dropout = 0.5
ponder_step = 5
concat = False
dot_str = ""

if concat:
  dot_str = "_concat"
else:
  dot_str = "_dot"

print_every = 200
load = False

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
inps, instrs, targets = data_loader.read_data("./dataset/lang_games_data_artificial_train_nvl_"
                                              + which_data + "_50000.txt")
inps_v, instrs_v, targets_v = data_loader.read_data("./dataset/lang_games_data_artificial_valid_nvl_"
                                                    + which_data + "_50000.txt")
inps_t, instrs_t, targets_t = data_loader.read_data("./dataset/lang_games_data_artificial_test_nvl_"
                                                    + which_data + "_50000.txt")
dataset = data_loader.Dataset(inps, instrs, targets, inps_v, instrs_v, targets_v, inps_t, instrs_t, targets_t)
dataset.randomize_data()

# initialize model
conf = [[32,64,128,256],[1,2],[0,0.2,0.5]]
config = list(itertools.product(*conf))
#config = [[128,2,0.5]]

def run_train(config):
  for j, elem in enumerate(config):
    model_name = "LSTM_50000_nvl_" + which_data + "_hid" + str(elem[0]) + \
                 "_layer" + str(elem[1]) + "_drop" + str(elem[2]) + dot_str
    batch_size = 200
    if load:
      decoder = torch.load("models/" + model_name + ".tar")
    else:
      decoder = Decoder.Decoder(dataset.n_letters, elem[0], dataset.n_letters, n_layers=elem[1],
                        dropout_p=elem[2], example_len=dataset.len_instr, concat=False)
    encoder = Encoder.Encoder(dataset.n_words, elem[0], n_layers=elem[1])
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    decoder.cuda()
    encoder.cuda()
    start = time.time()
    iters = 0
    decoder.train(True)
    encoder.train(True)
    losses, accs, accs_tr = [], [], []
    attn = True

    writer = SummaryWriter()

    cur_best = 0
    for epoch in range(1, n_epochs + 1):
      steps = len(inps) / batch_size
      for i in range(int(steps)):
        start_index = i * batch_size
        inp, instr, target = dataset.generate_batch(start_index, batch_size, inps, instrs, targets)
        loss = train.train(encoder, decoder, optimizer, criterion, dataset.len_targets, batch_size,
                           inp, instr, target, attn=attn)
        writer.add_scalar('data/loss', loss, iters)
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
      if acc_val_seq > cur_best:
        cur_best = acc_val_seq
        print("Writing models at epoch {}".format(epoch))
        with open("models/" + model_name + ".tar", 'wb') as ckpt:
          torch.save(decoder, ckpt)
      #accs.append(acc_seq)
      #accs_tr.append(acc_tr_seq)
      print("Config {}, Loss {}, Test Accuracy {}, Train Accuracy {}, Val Accuracy {}, "
            "Test Seq Accuracy {}, Train Seq Accuracy {}, Val Seq Accuracy {}"
        .format(str(j), loss, acc, acc_tr, acc_val, acc_seq, acc_tr_seq, acc_val_seq))

    writer.export_scalars_to_json("./json/"+ model_name +".json")
    writer.close()

run_train(config)