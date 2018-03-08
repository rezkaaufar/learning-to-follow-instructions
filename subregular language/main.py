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

## helper function ##
def time_since(since):
  now = time.time()
  s = now - since
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

## main ##
# initialize dataset
inps, labels = data_loader.read_data("./dataset/subreg_train_10000_nvlex.txt")
inps_t, labels_t = data_loader.read_data("./dataset/subreg_test_10000_nvlex.txt")
dataset = data_loader.Dataset(inps, labels, inps_t, labels)
dataset.randomize_data()

# initialize model
if not load:
  rnn = RNN.RNN(dataset._n_letters, n_hidden, dataset._n_categories,
                dropout_p=dropout, grammar_len=grammar_len, n_k_factors=n_k_factors)
  rnn.cuda()
else:
  rnn = torch.load('./models/lstm_5000example_20000c_attn_ponder_nvlex.tar').cuda()

rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


start = time.time()
iters = 0
rnn.train(True)
losses, accs, accs_tr = [], [], []
attn = True
ponder = False
pool = True

writer = SummaryWriter()

for epoch in range(1, n_epochs + 1):
    start_index = 0
    steps = len(inps) / batch_size
    for i in range(int(steps)):
        start_index = i * batch_size
        inp, target = dataset.generate_batch(start_index, batch_size, train=True)
        loss, _, _ = train.train(inp, target, rnn, rnn_optimizer, criterion, batch_size, len_example,
                                 attn=attn, ponder=ponder, pool=pool)
        writer.add_scalar('data/loss', loss, iters)
        losses.append(loss)
        if iters % print_every == 0:
            acc = evaluation.accuracy_test_data(dataset, rnn, batch_size, len_example, attn=attn, ponder=ponder, pool=pool)
            acc_tr = evaluation.accuracy_train_data(dataset, rnn, batch_size, len_example, attn=attn, ponder=ponder, pool=pool)
            accs.append(acc)
            accs_tr.append(acc_tr)
            writer.add_scalar('data/test_accuracy', acc, iters)
            writer.add_scalar('data/train_accuracy', acc_tr, iters)
            print("Loss {}, Test Accuracy {}, Train Accuracy {}".format(loss, acc, acc_tr))
            #print("Loss {}".format(loss))

        iters += 1

acc = evaluation.accuracy_test_data(dataset, rnn, batch_size, len_example, attn=attn, ponder=ponder, pool=pool)
acc_tr = evaluation.accuracy_train_data(dataset, rnn, batch_size, len_example, attn=attn, ponder=ponder, pool=pool)
accs.append(acc)
accs_tr.append(acc_tr)

writer.export_scalars_to_json("./all_scalars.json")
writer.close()

with open('./models/lstm_10000example_20000c_attn_pool_nvlex.tar', 'wb') as ckpt:
  torch.save(rnn, ckpt)