import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import Decoder
import data_loader
import train
import evaluation
import math
import itertools
from tensorboardX import SummaryWriter

## hyperparameters ##
# hyperparameters
n_epochs = 1
n_hidden = 128 # (32, 64, 128, 256)
n_layers = 2 # (1, 2)
lr = 1e-3
clip = 0.25
batch_size = 100
dropout = 0.5
ponder_step = 5

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
inps, targets = data_loader.read_data("./dataset/lang_games_data_artificial_train_nvl_utter_blocks_24000.txt")
inps_v, targets_v = data_loader.read_data("./dataset/lang_games_data_artificial_valid_nvl_utter_blocks_24000.txt")
inps_t, targets_t = data_loader.read_data("./dataset/lang_games_data_artificial_test_nvl_utter_blocks_24000.txt")
dataset = data_loader.Dataset(inps, targets, inps_v, targets_v, inps_t, targets_t)
dataset.randomize_data()

# initialize model
conf = [[32,64,128,256],[1,2],[0,0.2,0.5]]
config = list(itertools.product(*conf))

def run_train(config):
  for j, elem in enumerate(config):
    batch_size = 200
    decoder = Decoder.Decoder(dataset.n_letters, elem[0], dataset.n_letters, n_layers=elem[1],
                      dropout_p=elem[2], example_len=dataset.len_example, ponder_step=ponder_step)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    decoder.cuda()
    start = time.time()
    iters = 0
    decoder.train(True)
    losses, accs, accs_tr = [], [], []
    attn = True
    ponder = False

    writer = SummaryWriter()

    for epoch in range(1, n_epochs + 1):
      steps = len(inps) / batch_size
      for i in range(int(steps)):
        start_index = i * batch_size
        inp, target = dataset.generate_batch(start_index, dataset.len_example, batch_size, inps, targets)
        loss = train.train(decoder, optimizer, criterion, dataset.len_targets, dataset.all_characters,
                           batch_size, inp, target, attn=attn, ponder=ponder)
        writer.add_scalar('data/loss', loss, iters)
        losses.append(loss)
        iters += 1
      acc, acc_seq = evaluation.accuracy_test_data(dataset, decoder, inps_t, targets_t, attn=attn, ponder=ponder)
      acc_val, acc_val_seq = evaluation.accuracy_test_data(dataset, decoder, inps_v, targets_v, attn=attn, ponder=ponder)
      acc_tr, acc_tr_seq = evaluation.accuracy_train_data(dataset, decoder, inps, targets, batch_size, attn=attn, ponder=ponder)
      writer.add_scalar('data/test_accuracy', acc, iters)
      writer.add_scalar('data/train_accuracy', acc_tr, iters)
      writer.add_scalar('data/test_seq_accuracy', acc_seq, iters)
      writer.add_scalar('data/train_seq_accuracy', acc_tr_seq, iters)
      writer.add_scalar('data/val_accuracy', acc_val, iters)
      writer.add_scalar('data/val_seq_accuracy', acc_val_seq, iters)
      #accs.append(acc_seq)
      #accs_tr.append(acc_tr_seq)
      print("Config {}, Loss {}, Test Accuracy {}, Train Accuracy {}, Val Accuracy {}, "
            "Test Seq Accuracy {}, Train Seq Accuracy {}, Val Seq Accuracy {}"
        .format(str(j), loss, acc, acc_tr, acc_val, acc_seq, acc_tr_seq, acc_val_seq))

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    model_name = "LSTM_24000_nvl_utter_blocks_hid" + str(elem[0]) + "_layer" + str(elem[1]) + "_drop" + str(elem[2])
    with open("models/" + model_name + ".tar", 'wb') as ckpt:
      torch.save(decoder, ckpt)

run_train(config)