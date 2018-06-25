#!/usr/bin/env bash
#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=72:00:00

module load python/3.5.0
module load eb
module load CUDA

# human data #
#python3 $HOME/language_games/online_train.py --optim $OPTIM --lamb 0 --steps $STEPS --regularize $REGU --lr $LR --unfreezed 7 --k 1 --learner "gd" --output "hyperparameter_artificial_data_LSTM" --data "lang_games_data_artificial_train_online_nvl" &
#python3 $HOME/language_games/online_train.py --optim "Adam" --lamb 0 --steps 500 --regularize 0.001 --lr 0.01 --unfreezed 1 --k 7 --learner "gd" --output "recover_words_seq2conv_trial2" --data $DATA &
#python3 $HOME/language_games/online_train.py --optim "Adam" --lamb 0 --steps 500 --regularize 0.0 --lr 0.001 --unfreezed 7 --k 7 --learner "gd" --output "recover_words_seq2conv_trial2/un7" --data $DATA &
#python3 $HOME/language_games/online_train.py --optim "Adam" --lamb 0 --steps 500 --regularize 0.0 --lr 0.0001 --unfreezed 5 --k 7 --learner "gd" --output "recover_words_seq2conv_trial2/un5" --data $DATA &
#python3 $HOME/language_games/online_train.py --optim "Adam" --lamb 0 --steps 500 --regularize 0.001 --lr 0.01 --unfreezed 1 --k 1 --learner "gd" --output "recover_words_seq2conv_trial2/k_1" --data $DATA &

# conv2seq #
#python3 $HOME/language_games/conv2seq/main.py --hidden_size $HS --dropout_rate $DR --layers_conv $LC --layers_lstm $LL &

# seq2conv #
python3 $HOME/language_games/main.py --hidden_size $HS --dropout_rate $DR --layers_conv $LC --layers_lstm $LL &

# conv2conv #
#python3 $HOME/language_games/conv2conv/main.py --hidden_size $HS --dropout_rate $DR --layers_conv $LC &

# seq2seq #
#python3 $HOME/language_games/seq2seq/main.py --hidden_size $HS --dropout_rate $DR --layers_lstm $LC &

wait