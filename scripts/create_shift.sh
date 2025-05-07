#!/bin/bash

# ASSUMES YOU ALREADY HAVE:
# ../data/graphs/big_aug_gor_5e5.txt
# ../data/graphs/big_aug_gor_1e6.txt
# ../data/graphs/big_aug_gor_2e6.txt
# ../data/graphs/big_aug_gor_5e6.txt
# ../data/graphs/big_aug_gor_1e7.txt
# ../data/graphs/big_aug_gor_2e7.txt

# for loop where a variable iterates over specific strings that I set
for size in 5e5 1e6 2e6 5e6 1e7 2e7; do
    echo "Creating backups for $size"
    mv ../data/graphs/big_aug_gor_$size.txt ../data/graphs/big_aug_gor_$size\_backup.txt
    mv ../data/graphs/big_aug_gor_$size\_sorted.txt ../data/graphs/big_aug_gor_$size\_sorted_backup.txt
done


# insert your commands here
for ((b=2; b<=5; b++)); do
    echo "Creating copy $b"
    for sizes in "5 5" "1 6" "2 6" "5 6" "1 7" "2 7"; do
        size=( $sizes )
        newgraph=../data/graphs/big_aug_gor_${size[0]}e${size[1]}\_v$b.txt
        if [ ! -f "${newgraph}" ]; then
            bash big_graph_creator.sh ${size[0]} ${size[1]} gor 6
            mv ../data/graphs/big_aug_gor_${size[0]}e${size[1]}.txt ../data/graphs/big_aug_gor_${size[0]}e${size[1]}\_v$b.txt
            # mv ../data/graphs/big_aug_gor_${size[0]}e${size[1]}\_sorted.txt ../data/graphs/big_aug_gor_${size[0]}e${size[1]}\_sorted_v$b.txt
            rm ../data/graphs/big_aug_gor_${size[0]}e${size[1]}\_sorted.txt
            rm ../data/graphs/big_aug_gor_${size[0]}e${size[1]}\_bare.txt
        else
            echo "../data/graphs/big_aug_gor_${size[0]}e${size[1]}\_v$b.txt already exists, skipping..."
        fi
    done | parallel -j 4 {}
done


# for loop where a variable iterates over specific strings that I set
for size in 5e5 1e6 2e6 5e6 1e7 2e7; do
    echo "Recovering from backups for $size"
    mv ../data/graphs/big_aug_gor_$size\_backup.txt ../data/graphs/big_aug_gor_$size.txt
    mv ../data/graphs/big_aug_gor_$size\_sorted_backup.txt ../data/graphs/big_aug_gor_$size\_sorted.txt
done

cat create_shift_from_aug.txt | parallel -j 4 {}





