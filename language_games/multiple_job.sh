#!/bin/bash
declare -a arr=("AB9OKOQAD8OSP"
"AK3H5QRAROFGP"
"A2AWBDW9V1E9KT"
"A1F669OTXWIJW0"
"A1XVASF3XZ9FDP"
"AOXM7IK32TST5"
"A3SDFAOJW87FGP"
"A1U5BE8XJRXKW3"
"A2F9V69F6TZIAB"
"AJD3JOWNVDZD1"
"A1KK5PS8X2U5QP"
"A1M682B2WUSYJP"
"A1SWLR0UPOFRCW"
"A1W5D2MRLBNBJU"
"A2MUR0MDDWK4TY"
"ASOOONBGQ48YI"
"A1JNC3HWWHJD2J"
"A2J9AHXZBB37XU"
"A8L450UGY77XB"
"A1VL7507UONPX0"
"A34D527JHR5NS8"
"A3LZCR1FDVSVQ8"
"A1VI3SOIHT6Y0D"
"A9HQ3E0F2AGVO"
"A3JI3B5GTVA95F"
"A2DNGVGCC5K8TS"
"A17AMTRIYI2Z2D"
"A2H5V1V8QIY8Y7"
"ATMQRDBW7WM1Z"
"A2QKPBMYWR4KT3"
"AS7WV8YWOEO55"
"A34QARS9MSTJ08"
"A1FBBY2JJYRMRI"
"A2YJFCTJPPX66"
"A1XDMS0KFSF5JW"
"A25KGFC9X1OAOJ"
"A2S3MXU8T2I5H0"
"A2YKW761AK4ZGY"
"A21FSF2MCK11IR"
"A1A3TGZ7DKJWRW"
"A3IA77UHAVA83X"
"A341XKSRZ58FJK"
"A2SD7GN80U31GW"
"A1VLZL3CA1WMPT"
"A3MPHGI584PR1U"
"A3VRDQJNM1IC0E"
"A3SO3B7PJ4N6UQ"
"AFDC9A6Z60W2Z"
"A19V9DLZS91SDL"
"A3SM6R11R3687Q"
"A2AMI7BVALOCJP"
"A2YJQUHFIXNOY0"
"A27SJIOU39S074"
"A610SH5RY1NG1"
"AU5UJIZ3TLI5M"
"A1OK974KXKO6FJ"
"A19XAU8SBQZLYM"
"A8SD5DJ3KBLLR"
"AKLV0WIZZ356X"
"AHXDKQOG8YHLP"
"A2Z70GL7HTFFQR"
"A1TH0PTGDSBWMO"
"AGBDM85RLYTGL"
"AIEKCWYZTS41V"
"A3SHP9IPZ2FLOX"
"A3ON3137RY8LPM"
"A3UV55HC87DO9C"
"A3RR85PK3AV9TU"
"ASS4LIVDW452F"
"A3OYUJ6E6BJS4H"
"A40LU73SS18IW"
"A39W0L8ENJR734"
"A5TI9UQEPLNWI"
"A3HNEYFOIJWPH1"
"A2TT6FAIWFD25N"
"A98XHW6B1VSSQ"
"A27890V91M6OO9"
"A3QW5NQC6ZBXJ4"
"A39QOA9M7GNF86"
"AXPP6I0W11RTS"
"AJ6IIWWL0KF5Q"
"A3N5L136KKW6AV"
"A3UNS38BYD3OQR"
"A22M4NS65GY000"
"A1C6BL1O5L5LA9"
"A2AK6ARIIOZ2VE"
"A2C84QVRK3KG57"
"A1XH05IKC77OXO"
"A153FVJRRM27G7"
"AX6JQ37WUHFSH"
"A10G57EYF3HK46"
"A3EHI7J2E5DSJK"
"A1IU5OP7BBZHZ7"
"A1TL8OAM2V1Q5K"
"AND381S2NK11X"
"A9E9L9MWEZOPN")

#"remove_brown_every"

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

#for HS in 256; do
#	for DR in 0.5; do
#	    for LC in 5; do
#	        for LL in 2; do
#	            for W in "reorder" "masked"; do
# 		            qsub -v HS=$HS,DR=$DR,LC=$LC,LL=$LL,W=$W -q gpu $HOME/language_games/job.sh
# 		        done
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
# 		        qsub -v HS=$HS,DR=$DR,LC=$LC -q gpu $HOME/language_games/job.sh
# 		done
#	done
#done

# human data artificial #
for i in "${arr[@]}"; do
    qsub -v DATA=$i -q gpu $HOME/language_games/job.sh
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