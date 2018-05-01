import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import Decoder
import Encoder
import data_loader
import helper_online_train as hot
import train
import evaluation
import random
import math
import itertools
import numpy as np
import copy
from tensorboardX import SummaryWriter

## hyperparameters ##
# hyperparameters
n_epochs = 100
n_hidden = 64 # (32, 64, 128, 256)
n_layers = 1 # (1, 2)
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

## initialize dataset ##
which_data = "utter_blocks"
words_to_replace = ["add", "red", "orange", "1st", "3rd", "5th", "even"]
words_replacement = ["et", "roze", "oranje", "1", "3", "5", "ev"]
number_train = 20

## main ##
inps, instrs, targets = data_loader.read_data(cloud_str + "dataset/lang_games_data_artificial_train_nvl_"
                                              + which_data + "_50000.txt")
inps_v, instrs_v, targets_v = data_loader.read_data(cloud_str + "dataset/lang_games_data_artificial_valid_nvl_"
                                                    + which_data + "_50000.txt")
inps_t, instrs_t, targets_t = data_loader.read_data(cloud_str + "dataset/lang_games_data_artificial_test_nvl_"
                                                    + which_data + "_50000.txt")
#inps_m, instrs_m, targets_m = hot.read_merged_data(cloud_str + "dataset/sida wang's/txt/A341XKSRZ58FJK.txt")
inps_m, instrs_m, targets_m = hot.read_merged_data(cloud_str + "dataset/lang_games_data_artificial_train_online_nvl.txt")

dataset = data_loader.Dataset(inps, instrs, targets, inps_v, instrs_v, targets_v, inps_t, instrs_t, targets_t)
dataset.randomize_data()

# add new vocab
wtr = []
for instr in instrs_m:
  cc = instr.split(" ")
  for el in cc:
    if el not in dataset.all_words:
      wtr.append(el)

all_words_ext = sorted(list(wtr))
n_words_ext = len(all_words_ext)
all_words_comb = copy.deepcopy(dataset.all_words)
for el in wtr:
  all_words_comb.append(el)
n_words_comb = len(all_words_comb)

## main run ##

def run_train_optim(num_init, optimizer, lamb, training_updates, learning_rate):
  loss_thres = 100000000 # some big numbers
  loss_thres_cv = 100000000  # some big numbers

  batch_size = 1

  predicted_at_ts = []

  # greedy #
  online_accuracy_best = 0
  model_losses = []
  online_accuracies = []

  # 1 out cv #
  online_accuracy_best_cv = 0
  model_losses_cv = []
  online_accuracies_cv = []

  # test acc #
  # acc_test_seq_best = 0
  # acc_test_seq_best_cv = 0
  # highest_test = 0
  # accs_test_seq = []

  # online acc rest #
  # highest_oa_rst = 0
  # oa_rsts = []

  for k in range(num_init):
    enc_ext = hot.EncoderExtended(dataset.n_words, n_words_ext, n_hidden, n_layers=n_layers)
    enc_ext.cuda()
    torch.nn.Module.dump_patches = True
    decoder = Decoder.ConvDecoder(dataset.n_letters, n_hidden, n_hidden, dataset.n_letters, dataset.len_example,
                                  kernel_size=3, n_layers=layers_conv, dropout_p=0.5, example_len=dataset.len_instr)
    encoder = Encoder.EncoderWord(dataset.n_words, n_hidden, n_layers=n_layers)
    decoder.load_state_dict(
      torch.load(cloud_str + 'models/Params_Decoder_Seq2Conv_50000_nvl_utter_blocks_hid64_layer1_drop0.5_dot.tar'))
    encoder.load_state_dict(
      torch.load(cloud_str + 'models/Params_Encoder_Seq2Conv_50000_nvl_utter_blocks_hid64_layer1_drop0.5_dot.tar'))
    decoder.cuda()
    encoder.cuda()
    criterion = nn.NLLLoss()
    # load parameters from trained model
    params1 = encoder.named_parameters()
    params2 = enc_ext.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
      if name1 in dict_params2:
        dict_params2[name1].data.copy_(param1.data)

    # freeze all weights except the connected one
    for name, params in enc_ext.named_parameters():
      if name != "embedding_ext.weight" and name != "lamb" and name != "embedding.weight":
        params.requires_grad = False

    if k==0:
      best_params = dict_params2["embedding_ext.weight"]
    if optimizer=="Adam":
      enc_ext_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, enc_ext.parameters()), lr=learning_rate)
    else:
      enc_ext_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, enc_ext.parameters()), lr=learning_rate)

    steps = len(inps_m) / batch_size
    iters = 0
    num_train_per_step = 1

    loss_eval = 0
    loss_eval_cv1 = 0
    online_accuracy = 0
    # online_accuracy_cv = 0
    model_loss = []
    model_loss_cv = []
    predicted_at_t = []
    # online_accuracy_rst = 0

    ### train and evaluate ###
    for i in range(int(steps)):

      ### [CV] evaluate on current online training instances and test accuracy ###
      start_index = i * batch_size
      len_ex = len(inps_m[i].split(" "))
      len_ins = len(instrs_m[i].split(" "))
      len_tgt = len(targets_m[i].split(" "))
      inp, instr, target = hot.generate_batch_ext(dataset, start_index, len_ex, len_tgt,
                                            len_ins, all_words_comb, batch_size, inps_m, instrs_m, targets_m)
      cv1_loss = hot.train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, len_ex, len_tgt,
                                            len_ins, n_hidden, batch_size, lamb,
                                            inp, instr, target, attn=attn, eval=True)
      loss_eval_cv1 += cv1_loss
      model_loss_cv.append(loss_eval_cv1)

      acc, acc_seq = hot.accuracy_test_data_ext(dataset, len_ex, len_tgt,
                                            len_ins, enc_ext, decoder, [inps_m[i]], [instrs_m[i]], [targets_m[i]],
                                            all_words_comb, n_hidden, batch_size, lamb, attn=attn)
      online_accuracy += acc_seq
      predicted_at_t.append(acc_seq)
      ### [CV END] ###

      ### TRAINING REGIME ###
      enc_ext.train(True)
      buf_range = i+1
      current_buf = list(range(buf_range))
      if i < num_train_per_step:
        inp_buffer = Variable(torch.zeros(buf_range, dataset.len_example).long()).cuda()
        lab_buffer = Variable(torch.zeros(buf_range, dataset.len_targets).long()).cuda()
        ins_buffer = Variable(torch.zeros(buf_range, dataset.len_instr).long()).cuda()
        for epoch in range(1, training_updates + 1):
          train_pick = random.sample(current_buf, buf_range)
          for j, tp in enumerate(train_pick):
            start_index = tp * batch_size
            # dataset, start_index, len_example, len_labels, len_instr, all_words_comb,
            # batch_size, inps, instrs, labels)
            inp, instr, target = hot.generate_batch_ext(dataset, start_index, len_ex, len_tgt,
                                            len_ins, all_words_comb, batch_size, inps_m, instrs_m, targets_m)
            inp_buffer[j] = inp
            ins_buffer[j] = instr
            lab_buffer[j] = target
          loss = hot.train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, len_ex, len_tgt,
                               len_ins, n_hidden, batch_size, lamb,
                               inp_buffer, ins_buffer, lab_buffer, attn=attn)
          loss_eval += loss
          iters += 1
        model_loss.append(loss_eval)
      else:
        inp_buffer = Variable(torch.zeros(num_train_per_step, dataset.len_example).long()).cuda()
        lab_buffer = Variable(torch.zeros(num_train_per_step, dataset.len_targets).long()).cuda()
        ins_buffer = Variable(torch.zeros(num_train_per_step, dataset.len_instr).long()).cuda()
        for epoch in range(1, n_epochs + 1):
          train_pick = random.sample(current_buf, num_train_per_step)
          for j, tp in enumerate(train_pick):
            start_index = tp * batch_size
            # dataset, start_index, len_example, len_labels, len_instr, all_words_comb,
            # batch_size, inps, instrs, labels)
            inp, instr, target = hot.generate_batch_ext(dataset, start_index, len_ex, len_tgt,
                                            len_ins, all_words_comb, batch_size, inps_m, instrs_m, targets_m)
            inp_buffer[j] = inp
            ins_buffer[j] = instr
            lab_buffer[j] = target
          loss = hot.train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, len_ex, len_tgt,
                               len_ins, n_hidden, batch_size, lamb,
                               inp_buffer, ins_buffer, lab_buffer, attn=attn)
          loss_eval += loss
          iters += 1
        model_loss.append(loss_eval)
      ### TRAINING END ###

      ### [Greedy] evaluate on current online training instances ###
      # acc, acc_seq = hot.accuracy_test_data_ext(dataset, enc_ext, decoder, [inps_m[i]], [instrs_m[i]], [targets_m[i]],
      #                                       all_words_comb, n_hidden, batch_size, lamb, attn=attn)
      # online_accuracy += acc_seq
      ### [Greedy] evaluate end ###

    ### append model ###
    model_losses.append(model_loss)
    model_losses_cv.append(model_loss_cv)
    predicted_at_ts.append(predicted_at_t)

    online_accuracy /= len(inps_m)
    # online_accuracy_cv /= len(inps_m)
    # acc, acc_seq = hot.accuracy_test_data_ext(dataset, enc_ext, decoder, inps_rst, instrs_rst, targets_rst,
    #                                       all_words_comb, n_hidden, batch_size, lamb, attn=attn)
    # online_accuracy_rst = acc_seq

    ### evaluate test accuracy ###
    # acc, acc_test_seq = hot.accuracy_test_data_ext(dataset, enc_ext, decoder, inps_mt, instrs_mt, targets_mt,
    #                                            all_words_comb, n_hidden, batch_size, lamb, attn=attn)
    # print("Accuracy on the Test Set {}".format(acc_test_seq))
    # print("Online Accuracy {}".format(online_accuracy))
    # print("Online Accuracy CV {}".format(online_accuracy_cv))
    # print("Online Accuracy Rest {}".format(online_accuracy_rst))
    online_accuracies.append(online_accuracy)
    online_accuracies_cv.append(online_accuracy)
    # oa_rsts.append(online_accuracy_rst)
    # accs_test_seq.append(acc_test_seq)

    ### [Greedy] retrieve the best params and best online_accuracy ###
    if loss_eval < loss_thres:
      loss_thres = loss_eval
      online_accuracy_best = online_accuracy
      # acc_test_seq_best = acc_test_seq

    ### [1-out CV] retrieve the best params and best online_accuracy ###
    if loss_eval_cv1 < loss_thres_cv:
      loss_thres_cv = loss_eval_cv1
      online_accuracy_best_cv = online_accuracy
      # acc_test_seq_best_cv = acc_test_seq

    ### highest test and online rest acc ###
    # if acc_test_seq > highest_test:
    #   highest_test = acc_test_seq
    # if online_accuracy_rst > highest_oa_rst:
    #   highest_oa_rst = online_accuracy_rst

  ### final result ###
  #print("Final Online Accuracy {}".format(online_accuracy_best))
  # print("Final Test Seq Accuracy {}".format(acc_test_seq_best))
  #print("Final Online Accuracy CV {}".format(online_accuracy_best_cv))
  # print("Final Test Seq Accuracy CV {}".format(acc_test_seq_best_cv))
  # print("Highest Test Seq Accuracy {}".format(highest_test))
  # print("Highest Online Accuracy Rest {}".format(highest_oa_rst))
  ml = np.array(model_losses)
  # oa = np.array(online_accuracies)
  # oarst = np.array(oa_rsts)
  mlcv = np.array(model_losses_cv)
  # oacv = np.array(online_accuracies_cv)
  # ats = np.array(accs_test_seq)

  #print("Picked Model Greedy {}".format(np.argmin(ml,axis=0).tolist()))
  #print("Model ID with highest online accuracy train {}".format(np.argmax(oa)))
  #print("Picked Model 1CV {}".format(np.argmin(mlcv, axis=0).tolist()))
  #print("Model ID with highest online accuracy train CV {}".format(np.argmax(oacv)))
  # print("Model ID with highest test accuracy {}".format(np.argmax(ats)))
  # print("Model ID with highest online accuracy rest {}".format(np.argmax(oarst)))
  mls = np.argmin(ml, axis=0).tolist()
  mlcvs = np.argmin(mlcv, axis=0).tolist()
  pats = np.array(predicted_at_ts)
  res_ml = []
  for i, el in enumerate(mls):
    res_ml.append(pats[el,i])
  res_mlcv = []
  for i, el in enumerate(mlcvs):
    res_mlcv.append(pats[el,i])
  print(pats)
  # [fin online acc greedy, fin online acc 1cv, picked model greedy, model id highest greedy, picked model 1cv, model id highest 1cv]
  return [np.mean(res_ml), np.mean(res_mlcv), res_ml, res_mlcv, mls, mlcvs]

def run_random_search(k_trial, lamb):
  batch_size = 1
  avg_acc_test, avg_online = 0, 0
  max_avg, max_avg_online = 0, 0
  predicted_at_ts = []
  for k in range(k_trial):
    enc_ext = hot.EncoderExtended(dataset.n_words, n_words_ext, n_hidden, n_layers=n_layers)
    enc_ext.cuda()
    torch.nn.Module.dump_patches = True
    decoder = Decoder.ConvDecoder(dataset.n_letters, n_hidden, n_hidden, dataset.n_letters, dataset.len_example,
                                  kernel_size=3, n_layers=layers_conv, dropout_p=0.5, example_len=dataset.len_instr)
    encoder = Encoder.EncoderWord(dataset.n_words, n_hidden, n_layers=n_layers)
    decoder.load_state_dict(torch.load(cloud_str + 'models/Params_Decoder_Seq2Conv_50000_nvl_utter_blocks_hid64_layer1_drop0.5_dot_new.tar'))
    encoder.load_state_dict(torch.load(cloud_str + 'models/Params_Encoder_Seq2Conv_50000_nvl_utter_blocks_hid64_layer1_drop0.5_dot_new.tar'))
    decoder.cuda()
    encoder.cuda()
    criterion = nn.NLLLoss()
    # load parameters from trained model
    params1 = encoder.named_parameters()
    params2 = enc_ext.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
      if name1 in dict_params2:
        dict_params2[name1].data.copy_(param1.data)

    # freeze all weights except the connected one
    for name, params in enc_ext.named_parameters():
      if name != "embedding_ext.weight" and name != "lamb":
        params.requires_grad = False

    enc_ext_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, enc_ext.parameters()), lr=1e-3)

    # inps_new = inps_m + inps_rst
    # instrs_new = instrs_m + instrs_rst
    # targets_new = targets_m + targets_rst

    steps = len(inps_m) / batch_size
    loss_threshold = 1e-4
    online_accuracy = 0
    predicted_at_t = []
    ### train ###

    dict_params2["embedding_ext.weight"].data.normal_(0.0, 2.0)
    # torch.nn.init.xavier_normal(dict_params2["embedding_ext.weight"])
    dict_params2["lamb"].data = torch.rand(1).cuda()

    for i in range(int(steps)):
      start_index = i * batch_size
      len_ex = len(inps_m[i].split(" "))
      len_ins = len(instrs_m[i].split(" "))
      len_tgt = len(targets_m[i].split(" "))
      inp, instr, target = hot.generate_batch_ext(dataset, start_index, len_ex, len_tgt,
                                              len_ins, all_words_comb, batch_size, inps_m, instrs_m,
                                              targets_m)
      acc, acc_seq = hot.accuracy_test_data_ext(dataset, len_ex, len_tgt, len_ins,
                                                enc_ext, decoder, [inps_m[i]], [instrs_m[i]], [targets_m[i]],
                                                all_words_comb, n_hidden, batch_size, lamb, attn=attn)
      #print("Data : {}".format(inps_new[i] + " " + instrs_new[i] + " " + targets_new[i]))
      eval_loss = hot.train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, len_ex, len_tgt, len_ins,
                                n_hidden, batch_size, lamb,
                           inp, instr, target, attn=attn, eval=True)
      #print("Accuracy on " + str(i) + "th data : {}, Loss {}".format(acc_seq, eval_loss))
      predicted_at_t.append(acc_seq)
      online_accuracy += acc_seq
      # if acc_seq < 1:
      #   trial = 0
      #   while eval_loss > loss_threshold and trial < 2:
      #     dict_params2["embedding_ext.weight"].data.normal_(0.0, 2.0)
      #     #torch.nn.init.xavier_normal(dict_params2["embedding_ext.weight"])
      #     dict_params2["lamb"].data = torch.rand(1).cuda()
      #     eval_loss = hot.train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, n_hidden, batch_size, lamb,
      #                          inp, instr, target, attn=attn, eval=True)
      #     trial += 1
    ### evaluate on the rest ###
    online_accuracy /= len(inps_m)
    # acc, acc_seq = hot.accuracy_test_data_ext(dataset, enc_ext, decoder, inps_rst, instrs_rst, targets_rst,
    #                                       all_words_comb, n_hidden, batch_size, lamb, attn=attn)
    # online_accuracy += acc_seq
    print("Online Accuracy {}".format(online_accuracy))
    # acc, acc_test_seq = hot.accuracy_test_data_ext(dataset, enc_ext, decoder, inps_mt, instrs_mt, targets_mt,
    #                                       all_words_comb, n_hidden, batch_size, lamb, attn=attn)
    # print("Accuracy on the Test Set {}".format(acc_test_seq))
    # avg_acc_test += acc_test_seq
    oacc = online_accuracy
    avg_online += online_accuracy
    # if max_avg < acc_test_seq:
    #   max_avg = acc_test_seq
    if max_avg_online < oacc:
      max_avg_online = oacc
    predicted_at_ts.append(predicted_at_t)
  print(predicted_at_ts)
  print("Max Test Acc Seq {} Max Online Acc {}".format(max_avg, max_avg_online))


### trial random ###
#run_random_search(10)

### trial greedy ###
#conf = [["Adam", "SGD"], [True, False],[5, 10, 20, 50],[1e-2, 1e-3, 1e-4, 1e-5]]
#conf = [["Adam"], [True],[20, 50, 100],[1e-2, 1e-3, 1e-4, 1e-5]]
#config = list(itertools.product(*conf))
config = [('Adam', True, 50, 1e-2), ('Adam', True, 50, 1e-3)]
k_model = 15

spec_name = ""
for el in words_to_replace:
  spec_name += el + "_"
spec_name = spec_name[:-1]

#f = open(cloud_str + "online-result/" + spec_name + ".txt", "w")
f = open(cloud_str + "online-result/A341XKSRZ58FJK.txt", "w")
for c in config:
  t_start = time.time()
  res = run_train_optim(k_model, c[0], c[1], c[2], c[3])
  hyper_comb = " ".join(str(z) for z in c)
  f.write(hyper_comb + "\n")
  f.write(str(res) + "\n")
  print(hot.time_since(t_start))
f.close()

### trial 1cv ###
#run_train_1cv(10)

#run_random_search(10, True)


### kodingan cadangan ###
# def run_train_1cv(num_init):
#   loss_thres = 100000000 # some big numbers
#   batch_size = 1
#   model_losses = []
#   online_accuracies = []
#   accs_test_seq = []
#   online_accuracy_best = 0
#   acc_test_seq_best = 0
#   highest_test = 0
#   for k in range(num_init):
#     model_loss = []
#     enc_ext = EncoderExtended(dataset.n_words, n_words_ext, n_hidden, n_layers=n_layers)
#     enc_ext.cuda()
#     torch.nn.Module.dump_patches = True
#     decoder = Decoder.ConvDecoder(dataset.n_letters, n_hidden, n_hidden, dataset.n_letters, dataset.len_example,
#                                   kernel_size=3, n_layers=layers_conv, dropout_p=0.5, example_len=dataset.len_instr)
#     encoder = Encoder.EncoderWord(dataset.n_words, n_hidden, n_layers=n_layers)
#     decoder.load_state_dict(
#       torch.load('models/Params_Decoder_Seq2Conv_50000_nvl_utter_blocks_hid64_layer1_drop0.5_dot.tar'))
#     encoder.load_state_dict(
#       torch.load('models/Params_Encoder_Seq2Conv_50000_nvl_utter_blocks_hid64_layer1_drop0.5_dot.tar'))
#     decoder.cuda()
#     encoder.cuda()
#     criterion = nn.NLLLoss()
#     # load parameters from trained model
#     params1 = encoder.named_parameters()
#     params2 = enc_ext.named_parameters()
#
#     dict_params2 = dict(params2)
#
#     for name1, param1 in params1:
#       if name1 in dict_params2:
#         dict_params2[name1].data.copy_(param1.data)
#
#     # freeze all weights except the connected one
#     for name, params in enc_ext.named_parameters():
#       if name != "embedding_ext.weight" and name != "lamb":
#         params.requires_grad = False
#
#     enc_ext_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, enc_ext.parameters()), lr=1e-3)
#
#     steps = len(inps_m) / batch_size
#     iters = 0
#     online_accuracy = 0
#     num_train_per_step = 1
#     loss_eval = 0
#
#     ### train and evaluate ###
#     for i in range(int(steps)):
#       ### evaluate on current online training instances ###
#       start_index = i * batch_size
#       inp, instr, target = generate_batch_ext(dataset, start_index, dataset.len_example, dataset.len_targets,
#                                               dataset.len_instr,
#                                               all_words_comb, batch_size, inps_m, instrs_m, targets_m)
#       cv1_loss = train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset.len_targets, batch_size, lamb,
#                            inp, instr, target, attn=attn, eval=True)
#       loss_eval += cv1_loss
#       model_loss.append(loss_eval)
#
#       enc_ext.train(True)
#       ### train with expanded example ###
#       buf_range = i+1
#       current_buf = list(range(buf_range))
#       if i < num_train_per_step:
#         inp_buffer = Variable(torch.zeros(buf_range, dataset.len_example).long()).cuda()
#         lab_buffer = Variable(torch.zeros(buf_range, dataset.len_targets).long()).cuda()
#         ins_buffer = Variable(torch.zeros(buf_range, dataset.len_instr).long()).cuda()
#         for epoch in range(1, n_epochs + 1):
#           train_pick = random.sample(current_buf, buf_range)
#           for j, tp in enumerate(train_pick):
#             start_index = tp * batch_size
#             # dataset, start_index, len_example, len_labels, len_instr, all_words_comb,
#             # batch_size, inps, instrs, labels)
#             inp, instr, target = generate_batch_ext(dataset, start_index, dataset.len_example, dataset.len_targets,
#                                                     dataset.len_instr,
#                                                     all_words_comb, batch_size, inps_m, instrs_m, targets_m)
#             inp_buffer[j] = inp
#             ins_buffer[j] = instr
#             lab_buffer[j] = target
#           _ = train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset.len_targets, batch_size, lamb,
#                              inp_buffer, ins_buffer, lab_buffer, attn=attn)
#           iters += 1
#       else:
#         inp_buffer = Variable(torch.zeros(num_train_per_step, dataset.len_example).long()).cuda()
#         lab_buffer = Variable(torch.zeros(num_train_per_step, dataset.len_targets).long()).cuda()
#         ins_buffer = Variable(torch.zeros(num_train_per_step, dataset.len_instr).long()).cuda()
#         for epoch in range(1, n_epochs + 1):
#           train_pick = random.sample(current_buf, num_train_per_step)
#           for j, tp in enumerate(train_pick):
#             start_index = tp * batch_size
#             # dataset, start_index, len_example, len_labels, len_instr, all_words_comb,
#             # batch_size, inps, instrs, labels)
#             inp, instr, target = generate_batch_ext(dataset, start_index, dataset.len_example, dataset.len_targets,
#                                                     dataset.len_instr,
#                                                     all_words_comb, batch_size, inps_m, instrs_m, targets_m)
#             inp_buffer[j] = inp
#             ins_buffer[j] = instr
#             lab_buffer[j] = target
#           _ = train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset.len_targets, batch_size, lamb,
#                         inp_buffer, ins_buffer, lab_buffer, attn=attn)
#           iters += 1
#
#       acc, acc_seq = accuracy_test_data_ext(dataset, enc_ext, decoder, [inps_m[i]], [instrs_m[i]], [targets_m[i]],
#                                             all_words_comb, batch_size, lamb, attn=attn)
#       # print("Data : {}".format(inps_m[i] + " " + instrs_m[i] + " " + targets_m[i]))
#       # print("Accuracy on " + str(i) + "th data : {}, Loss {}".format(acc_seq, loss_eval))
#       online_accuracy += acc_seq
#
#     ### evaluate on the rest ###
#     # for i in range(len(inps_rst)):
#       # start_index = i * batch_size
#       # inp, instr, target = generate_batch_ext(dataset, start_index, dataset.len_example, dataset.len_targets, dataset.len_instr,
#       #                                         all_words_comb, batch_size, inps_rst, instrs_rst, targets_rst)
#       # eval_loss = train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset.len_targets, batch_size, lamb,
#       #                  inp, instr, target, attn=attn, eval=True)
#       # acc, acc_seq = accuracy_test_data_ext(dataset, enc_ext, decoder, [inps_rst[i]], [instrs_rst[i]], [targets_rst[i]],
#       #                                       all_words_comb, batch_size, lamb, attn=attn)
#       # print("Eval Data : {}".format(inps_rst[i] + " " + instrs_rst[i] + " " + targets_rst[i]))
#       # print("Accuracy on " + str(i) + "th eval data : {}, Loss {}".format(acc_seq, eval_loss))
#       # online_accuracy += acc_seq
#     online_accuracy /= len(inps_m)
#     acc, acc_seq = accuracy_test_data_ext(dataset, enc_ext, decoder, inps_rst, instrs_rst, targets_rst,
#                                           all_words_comb, batch_size, lamb, attn=attn)
#     online_accuracy += acc_seq
#
#     ### evaluate test accuracy ###
#     acc, acc_test_seq = accuracy_test_data_ext(dataset, enc_ext, decoder, inps_mt, instrs_mt, targets_mt,
#                                                all_words_comb, batch_size, lamb, attn=attn)
#     print("Accuracy on the Test Set {}".format(acc_test_seq))
#     print("Online Accuracy {}".format(online_accuracy / 2))
#     model_losses.append(model_loss)
#     online_accuracies.append(online_accuracy / 2)
#     accs_test_seq.append(acc_test_seq)
#     if loss_eval < loss_thres:
#       loss_thres = loss_eval
#       online_accuracy_best = online_accuracy
#       acc_test_seq_best = acc_test_seq
#     if acc_test_seq > highest_test:
#       highest_test = acc_test_seq
#
#   ### final result ###
#   print("Final Online Accuracy {}".format(online_accuracy_best / 2))
#   print("Final Test Seq Accuracy {}".format(acc_test_seq_best))
#   print("Highest Test Seq Accuracy {}".format(highest_test))
#   ml = np.array(model_losses)
#   oa = np.array(online_accuracies)
#   ats = np.array(accs_test_seq)
#
#   print(np.argmin(ml,axis=0).tolist())
#   print(np.argmax(oa).tolist())
#   print(np.argmax(ats).tolist())