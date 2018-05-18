import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import Decoder
import encoder_attn
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
import os

## hyperparameters ##
# hyperparameters
n_epochs = 100
n_hidden = 256 # (32, 64, 128, 256)
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

dirs = os.path.dirname(os.path.abspath(__file__))
cur_decoder = "/models/Params_Decoder_Trans2Conv_50000_nvl_utter_blocks_hidden_size_256-dropout_rate_0.5-layers_conv_5-mean_attn_True.tar"
cur_encoder = "/models/Params_Encoder_Trans2Conv_50000_nvl_utter_blocks_hidden_size_256-dropout_rate_0.5-layers_conv_5-mean_attn_True.targt"


## main run ##

def run_train_optim(num_init, human_data, optimizer, lamb, training_updates, learning_rate, unfreezed=1, mean_attn=True):
  ## initialize dataset ##
  which_data = "utter_blocks"
  words_to_replace = ["add", "red", "orange", "1st", "3rd", "5th", "even"]
  words_replacement = ["et", "roze", "oranje", "1", "3", "5", "ev"]
  number_train = 20

  ## main ##
  inps, instrs, targets = data_loader.read_data(dirs + "/dataset/lang_games_data_artificial_train_nvl_"
                                                + which_data + "_50000.txt")
  inps_v, instrs_v, targets_v = data_loader.read_data(dirs + "/dataset/lang_games_data_artificial_valid_nvl_"
                                                      + which_data + "_50000.txt")
  inps_t, instrs_t, targets_t = data_loader.read_data(dirs + "/dataset/lang_games_data_artificial_test_nvl_"
                                                      + which_data + "_50000.txt")
  inps_m, instrs_m, targets_m = hot.read_merged_data(
    dirs + "/dataset/sida wang's/txt/" + human_data + ".txt")
  # inps_m, instrs_m, targets_m = hot.read_merged_data(
  #   dirs + "/dataset/online_test/" + human_data + ".txt")
  # inps_m, instrs_m, targets_m = hot.read_merged_data(dirs + "/dataset/" + human_data + ".txt")

  dataset = data_loader.Dataset(inps, instrs, targets, inps_v, instrs_v, targets_v, inps_t, instrs_t, targets_t)
  dataset.randomize_data()

  # add new vocab
  wtr = set()
  for instr in instrs_m:
    cc = instr.split(" ")
    for el in cc:
      if el not in dataset.all_words:
        wtr.add(el)

  all_words_ext = sorted(list(wtr))
  n_words_ext = len(all_words_ext)
  all_words_comb = copy.deepcopy(dataset.all_words)
  for el in wtr:
    all_words_comb.append(el)
  n_words_comb = len(all_words_comb)

  ##################################### MAIN RUN ####################################

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
    enc_ext = hot.Encoder2Extended(dataset.n_words, n_words_ext, n_hidden)
    enc_ext.cuda()
    torch.nn.Module.dump_patches = True
    decoder = Decoder.ConvDecoder(dataset.n_letters, n_hidden, n_hidden, dataset.n_letters, dataset.len_example,
                                  kernel_size=3, n_layers=layers_conv, dropout_p=0.5, example_len=dataset.len_instr)
    encoder = encoder_attn.Encoder(dataset.n_words, n_hidden)
    if unfreezed != 5:
      decoder.load_state_dict(
        torch.load(dirs + cur_decoder))
    encoder.load_state_dict(
      torch.load(dirs + cur_encoder))
    decoder.cuda()
    encoder.cuda()
    criterion = nn.NLLLoss()
    # load parameters from trained model
    params1 = encoder.named_parameters()
    params2 = enc_ext.named_parameters()

    dict_params2 = dict(params2)
    if unfreezed != 4:
      for name1, param1 in params1:
        if name1 in dict_params2:
          dict_params2[name1].data.copy_(param1.data)

      # freeze all weights except the connected one
      for name, params in enc_ext.named_parameters():
        if unfreezed == 1:
          if name != "embedding_ext.weight":
            params.requires_grad = False
        elif unfreezed == 2:
          if name != "embedding_ext.weight" and name != "embedding.weight":
            params.requires_grad = False

    if k==0:
      best_params = dict_params2["embedding_ext.weight"]
    if optimizer=="Adam":
      enc_ext_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, enc_ext.parameters()), lr=learning_rate)
      if unfreezed == 3 or unfreezed == 4 or unfreezed == 5:
        decoder_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate)
    else:
      enc_ext_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, enc_ext.parameters()), lr=learning_rate)
      if unfreezed == 3 or unfreezed == 4 or unfreezed == 5:
        decoder_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate)

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
      if unfreezed == 3 or unfreezed == 4 or unfreezed == 5:
        cv1_loss = hot.train_ext_2_unfreezed(enc_ext, decoder, enc_ext_optimizer, decoder_optimizer, criterion, dataset, len_ex, len_tgt,
                                              len_ins, n_hidden, batch_size,
                                              inp, instr, target, mean_attn, attn=attn, eval=True)
      else:
        cv1_loss = hot.train_ext_2(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, len_ex, len_tgt,
                                              len_ins, n_hidden, batch_size,
                                              inp, instr, target, mean_attn, attn=attn, eval=True)
      loss_eval_cv1 += cv1_loss
      model_loss_cv.append(loss_eval_cv1)

      acc, acc_seq = hot.accuracy_test_data_ext_2(dataset, len_ex, len_tgt,
                                            len_ins, enc_ext, decoder, [inps_m[i]], [instrs_m[i]], [targets_m[i]],
                                            all_words_comb, n_hidden, batch_size, mean_attn, attn=attn)
      online_accuracy += acc_seq
      predicted_at_t.append(acc_seq)
      ### [CV END] ###

      ### TRAINING REGIME ###
      enc_ext.train(True)
      buf_range = i+1
      current_buf = list(range(buf_range))
      if i < num_train_per_step:
        for epoch in range(1, training_updates + 1):
          train_pick = random.sample(current_buf, buf_range)
          len_ex = len(inps_m[train_pick[0]].split(" "))
          len_ins = len(instrs_m[train_pick[0]].split(" "))
          len_tgt = len(targets_m[train_pick[0]].split(" "))
          inp_buffer = Variable(torch.zeros(num_train_per_step, len_ex).long()).cuda()
          lab_buffer = Variable(torch.zeros(num_train_per_step, len_tgt).long()).cuda()
          ins_buffer = Variable(torch.zeros(num_train_per_step, len_ins).long()).cuda()
          for j, tp in enumerate(train_pick):
            start_index = tp * batch_size
            # dataset, start_index, len_example, len_labels, len_instr, all_words_comb,
            # batch_size, inps, instrs, labels)
            len_ex = len(inps_m[start_index].split(" "))
            len_ins = len(instrs_m[start_index].split(" "))
            len_tgt = len(targets_m[start_index].split(" "))
            inp, instr, target = hot.generate_batch_ext(dataset, start_index, len_ex, len_tgt,
                                            len_ins, all_words_comb, batch_size, inps_m, instrs_m, targets_m)
            inp_buffer[j] = inp
            ins_buffer[j] = instr
            lab_buffer[j] = target
          if unfreezed == 3 or unfreezed == 4 or unfreezed == 5:
            loss = hot.train_ext_2_unfreezed(enc_ext, decoder, enc_ext_optimizer, decoder_optimizer, criterion, dataset, len_ex, len_tgt,
                                 len_ins, n_hidden, batch_size,
                                 inp_buffer, ins_buffer, lab_buffer, mean_attn, attn=attn)
          else:
            loss = hot.train_ext_2(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, len_ex, len_tgt,
                                 len_ins, n_hidden, batch_size,
                                 inp_buffer, ins_buffer, lab_buffer, mean_attn, attn=attn)
          loss_eval += loss
          iters += 1
        model_loss.append(loss_eval)
      else:
        for epoch in range(1, training_updates + 1):
          train_pick = random.sample(current_buf, num_train_per_step)
          len_ex = len(inps_m[train_pick[0]].split(" "))
          len_ins = len(instrs_m[train_pick[0]].split(" "))
          len_tgt = len(targets_m[train_pick[0]].split(" "))
          inp_buffer = Variable(torch.zeros(num_train_per_step, len_ex).long()).cuda()
          lab_buffer = Variable(torch.zeros(num_train_per_step, len_tgt).long()).cuda()
          ins_buffer = Variable(torch.zeros(num_train_per_step, len_ins).long()).cuda()
          for j, tp in enumerate(train_pick):
            start_index = tp * batch_size
            # dataset, start_index, len_example, len_labels, len_instr, all_words_comb,
            # batch_size, inps, instrs, labels)
            inp, instr, target = hot.generate_batch_ext(dataset, start_index, len_ex, len_tgt,
                                            len_ins, all_words_comb, batch_size, inps_m, instrs_m, targets_m)
            inp_buffer[j] = inp
            ins_buffer[j] = instr
            lab_buffer[j] = target
          if unfreezed == 3 or unfreezed == 4 or unfreezed == 5:
            loss = hot.train_ext_2_unfreezed(enc_ext, decoder, enc_ext_optimizer, decoder_optimizer, criterion, dataset, len_ex, len_tgt,
                                 len_ins, n_hidden, batch_size,
                                 inp_buffer, ins_buffer, lab_buffer, mean_attn, attn=attn)
          else:
            loss = hot.train_ext_2(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, len_ex, len_tgt,
                                 len_ins, n_hidden, batch_size,
                                 inp_buffer, ins_buffer, lab_buffer, mean_attn, attn=attn)
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
  #print(pats)
  # [fin online acc greedy, fin online acc 1cv, picked model greedy, model id highest greedy, picked model 1cv, model id highest 1cv]
  return [np.mean(res_ml), np.mean(res_mlcv), res_ml, res_mlcv, mls, mlcvs]

def run_random_search(k_trial, human_data, lamb):
  ## initialize dataset ##
  which_data = "utter_blocks"
  words_to_replace = ["add", "red", "orange", "1st", "3rd", "5th", "even"]
  words_replacement = ["et", "roze", "oranje", "1", "3", "5", "ev"]
  number_train = 20

  ## main ##
  inps, instrs, targets = data_loader.read_data(dirs + "/dataset/lang_games_data_artificial_train_nvl_"
                                                + which_data + "_50000.txt")
  inps_v, instrs_v, targets_v = data_loader.read_data(dirs + "/dataset/lang_games_data_artificial_valid_nvl_"
                                                      + which_data + "_50000.txt")
  inps_t, instrs_t, targets_t = data_loader.read_data(dirs + "/dataset/lang_games_data_artificial_test_nvl_"
                                                      + which_data + "_50000.txt")
  inps_m, instrs_m, targets_m = hot.read_merged_data(
    dirs + "/dataset/sida wang's/txt/" + human_data + ".txt")
  # inps_m, instrs_m, targets_m = hot.read_merged_data(
  #   dirs + "/dataset/online_test/" + human_data + ".txt")
  # inps_m, instrs_m, targets_m = hot.read_merged_data(dirs + "/dataset/"+ human_data +".txt")

  dataset = data_loader.Dataset(inps, instrs, targets, inps_v, instrs_v, targets_v, inps_t, instrs_t, targets_t)
  dataset.randomize_data()

  # add new vocab
  wtr = set()
  for instr in instrs_m:
    cc = instr.split(" ")
    for el in cc:
      if el not in dataset.all_words:
        wtr.add(el)

  all_words_ext = sorted(list(wtr))
  n_words_ext = len(all_words_ext)
  all_words_comb = copy.deepcopy(dataset.all_words)
  for el in wtr:
    all_words_comb.append(el)
  n_words_comb = len(all_words_comb)

  ##################################### MAIN RUN ####################################

  batch_size = 1
  avg_acc_test, avg_online = 0, 0
  max_avg, max_avg_online = 0, 0
  predicted_at_ts = []
  for k in range(k_trial):
    enc_ext = hot.Encoder2Extended(dataset.n_words, n_words_ext, n_hidden)
    enc_ext.cuda()
    torch.nn.Module.dump_patches = True
    decoder = Decoder.ConvDecoder(dataset.n_letters, n_hidden, n_hidden, dataset.n_letters, dataset.len_example,
                                  kernel_size=3, n_layers=layers_conv, dropout_p=0.5, example_len=dataset.len_instr)
    encoder = encoder_attn.Encoder(dataset.n_words, n_hidden)
    decoder.load_state_dict(torch.load(dirs + cur_decoder))
    encoder.load_state_dict(torch.load(dirs + cur_encoder))
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
      if acc_seq < 1:
        trial = 0
        while eval_loss > loss_threshold and trial < 100:
          dict_params2["embedding_ext.weight"].data.normal_(0.0, 2.0)
          #torch.nn.init.xavier_normal(dict_params2["embedding_ext.weight"])
          dict_params2["lamb"].data = torch.rand(1).cuda()
          eval_loss = hot.train_ext(enc_ext, decoder, enc_ext_optimizer, criterion, dataset, len_ex, len_tgt, len_ins,
                                    n_hidden, batch_size, lamb,
                               inp, instr, target, attn=attn, eval=True)
          trial += 1
    ### evaluate on the rest ###
    online_accuracy /= len(inps_m)
    # acc, acc_seq = hot.accuracy_test_data_ext(dataset, enc_ext, decoder, inps_rst, instrs_rst, targets_rst,
    #                                       all_words_comb, n_hidden, batch_size, lamb, attn=attn)
    # online_accuracy += acc_seq
    # print("Online Accuracy {}".format(online_accuracy))
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
  #print(predicted_at_ts)
  #print("Max Test Acc Seq {} Max Online Acc {}".format(max_avg, max_avg_online))
  return max_avg_online

### trial random ###
#run_random_search(10)

### trial greedy ###
do_sweep = False
if do_sweep:
    #conf = [["Adam", "SGD"], [True, False],[5, 10, 20, 50, 100],[1e-2, 1e-3, 1e-4, 1e-5], [1,2,3]]
    #conf = [["Adam", "SGD"], [True, False], [5, 10, 20, 50, 100], [1e-2, 1e-3, 1e-4, 1e-5], [4]]
    #conf = [["Adam"], [True], [100], [1e-2], [1, 2, 3]]
    #conf = [["SGD"], [False],[50],[1e-5], [1,2,3]]
    #config = list(itertools.product(*conf))
    #config_rand = [True, False]
    config = [('Adam', True, 50, 1e-2, 3), ('Adam', False, 100, 1e-2, 3)]
    k_model = 7

    picked_human_data = ["AZGBKAM5JUV5A", "A1HKYY6XI2OHO1", "ADJ9I7ZBFYFH7"]

    # spec_name = ""
    # for el in words_to_replace:
    #   spec_name += el + "_"
    # spec_name = spec_name[:-1]

    #f = open(cloud_str + "online-result/" + spec_name + ".txt", "w")
    file = open(dirs + "/out.txt", "r")

    for human_data in picked_human_data:
      #human_data = line.replace("\n","")
      f = open(dirs + "/online-result/" + human_data + "unfreezed_2.txt", "w")
      for c in config:
        t_start = time.time()
        res = run_train_optim(k_model, human_data, c[0], c[1], c[2], c[3], c[4])
        hyper_comb = " ".join(str(z) for z in c)
        f.write(hyper_comb + "\n")
        f.write(str(res) + "\n")
        print(hot.time_since(t_start))
      # for c in config_rand:
      #   t_start = time.time()
      #   res = run_random_search(k_model+2, human_data, c)
      #   f.write("Random " + str(c) + "\n")
      #   f.write(str(res) + "\n")
      #   print(hot.time_since(t_start))
      f.close()
else:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--optim')
    ap.add_argument('--lamb', type=int)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--lr', type=float)
    ap.add_argument('--k', default=7, type=int)
    ap.add_argument('--unfreezed', type=int, choices=[1,2,3,4,5])
    ap.add_argument('--learner', choices=['random', 'gd'])
    ap.add_argument('--output', required=True)
    ap.add_argument('--mean_attn')
    ap.add_argument('--data')

    args =  ap.parse_args()

    fn = "-".join(["{}_{}".format(k, getattr(args, k)) for k in vars(args) if getattr(args, k) is not None])
    outdir = dirs + "/online-result/"+args.output
    os.makedirs(outdir, exist_ok=True)
    f = open(outdir +"/" + fn + ".txt", 'w')

    if args.lamb in ['yes', '1', 'true', 'True']:
        args.lamb = True
    else:
        args.lamb = False

    if args.mean_attn in ['yes', '1', 'true', 'True']:
        args.mean_attn = True
    else:
        args.mean_attn = False

    t_start = time.time()

    if args.learner == 'random':
        res = run_random_search(args.k+2, args.data, args.lamb)
        f.write("Random " + fn + "\n")
        f.write(str(res) + "\n")
    else:
        res = run_train_optim(args.k, args.data, args.optim, args.lamb, args.steps, args.lr, args.unfreezed, args.mean_attn)
        #hyper_comb = " ".join(str(z) for z in c)
        f.write(fn + "\n")
        f.write(str(res) + "\n")

    print(hot.time_since(t_start))
    f.close()
### trial 1cv ###
#run_train_1cv(10)

#run_random_search(10, True)
