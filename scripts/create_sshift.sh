#!/bin/bash

# ASSUMES YOU ALREADY HAVE:
# ../data/graphs/shift<b>_aug_gor_5e5.txt
# ../data/graphs/shift<b>_aug_gor_1e6.txt
# ../data/graphs/shift<b>_aug_gor_2e6.txt
# ../data/graphs/shift<b>_aug_gor_5e6.txt
# ../data/graphs/shift<b>_aug_gor_1e7.txt
# ../data/graphs/shift<b>_aug_gor_2e7.txt
# For b in 1 2 3 4 5

for ((b=1; b<=5; b++)); do
    echo "Creating scaling step graph $b"
    for sizes in "5 5" "1 6" "2 6" "5 6" "1 7" "2 7"; do
        size=( $sizes )
        coeff=${size[0]}
        pow=${size[1]}

        ../build/CreateGraph scaling_step ../data/graphs/shiftv${b}_aug_gor_${coeff}e${pow}.txt > ../data/graphs/sshiftv${b}_aug_gor_${size[0]}e${size[1]}.txt
    done
done