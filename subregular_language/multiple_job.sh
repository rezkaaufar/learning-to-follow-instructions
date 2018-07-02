#!/bin/bash

declare -a arr=("5"
"10"
"50"
"100"
"500"
"5000"
"10000")
#for j in "${arr[@]}"; do
#  #echo "$j"
#  qsub -q gpu -v PATH="$j" $HOME/language_games/job.sh
#  #qsub -v PATH="$j" $HOME/language_games/job.sh
#done

# conv2seq #
#for HS in 64; do
#	for DR in 0.2 0.5; do
#	    for LC in 4 5; do
#	        for LL in 1 2; do
# 		        qsub -v HS=$HS,DR=$DR,LC=$LC,LL=$LL -q gpu $HOME/language_games/job.sh
# 		    done
# 		done
#	done
#done

# seq2conv #
#for HS in 32 64 128 256; do
#	for DR in 0.2 0.5; do
#	    for LC in 4 5; do
#	        for LL in 1 2; do
# 		        qsub -v HS=$HS,DR=$DR,LC=$LC,LL=$LL -q gpu $HOME/language_games/job.sh
# 		    done
# 		done
#	done
#done

# conv2conv #
#for HS in 64 128 256; do
#	for DR in 0.2 0.5; do
#	    for LC in 4 5 6; do
# 		        qsub -v HS=$HS,DR=$DR,LC=$LC -q gpu $HOME/language_games/job.sh
# 		done
#	done
#done

# seq2seq #
#for HS in 32 64 128 256; do
#	for DR in 0.2 0.5; do
#	    for LC in 1 2; do
# 		        qsub -v HS=$HS,DR=$DR,LC=$LC -q gpu $HOME/subregular_language/job.sh
# 		done
#	done
#done

# human data artificial #
for i in "${arr[@]}"; do
    qsub -v DATA=$i -q gpu $HOME/subregular_language/job.sh
done

#for STEPS in 10 20 50 100 200 500 ; do
#	for OPTIM in "Adam" "SGD"; do
#	    for REGU in 0 1e-4 1e-3 1e-2; do
#	        for LR in 1e-1 1e-2 1e-3 1e-4 1e-5; do
# 		        qsub -v STEPS=$STEPS,OPTIM=$OPTIM,REGU=$REGU,LR=$LR -q gpu $HOME/language_games/job.sh
# 		    done
# 		done
#	done
#done