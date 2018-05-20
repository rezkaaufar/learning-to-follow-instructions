#!/usr/bin/env bash
#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=72:00:00

module load python/3.5.0
module load eb
module load CUDA

#Determining the number of processors in the system
#NPROC=`nproc --all`

#Execute program located in $HOME
#for i in `seq 1 $NPROC`; do
#  echo "$i"
#for j in "${arr[@]}"; do
python3 $HOME/language_games/online_train.py $PATH &
#python3 online_train.py $j &
#done
  #python3 $HOME/language_games/main.py &
  #python3 $HOME/language_games/online_train.py &
#done
wait