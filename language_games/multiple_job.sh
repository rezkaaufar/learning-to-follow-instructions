#!/bin/bash
declare -a arr=("--optim Adam --lamb 1 --regularize 0.0001 --steps 5 --lr 1e-05 --unfreezed 4 --k 7 --learner gd --output hyperparameter_artificial_data_LSTM --data lang_games_data_artificial_train_online_nvl"
"--optim Adam --lamb 1 --regularize 0.0001 --steps 5 --lr 1e-05 --unfreezed 5 --k 7 --learner gd --output hyperparameter_artificial_data_LSTM --data lang_games_data_artificial_train_online_nvl")

for j in "${arr[@]}"; do
  #echo "$j"
  qsub -q gpu -v PATH="$j" $HOME/language_games/job.sh
done

#for NUMBERS in 1 2 3 4 5; do
#	for LETTERS in a b c d e; do
# 		 qsub -v NUMBERARG=$NUMBERS,LETTERARG=$LETTERS my_qsub_script.pbs
#	done
#done