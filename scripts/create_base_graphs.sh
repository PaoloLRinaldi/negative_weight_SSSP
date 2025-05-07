#!/bin/bash

for sizes in "5 5" "1 6" "2 6" "5 6" "1 7" "2 7"; do
    size=( $sizes )
    coeff=${size[0]}
    pow=${size[1]}

    for alg in "bfct" "rd1" "rd2" "dfs" "gor"; do

        if [ ! -f "../data/graphs/big_aug_${alg}_${coeff}e${pow}.txt" ]; then
            bash big_graph_creator.sh $coeff $pow ${alg} 6

            # Delete temporary files
            rm ../data/graphs/big_aug_${alg}_${coeff}e${pow}_bare.txt
            rm ../data/graphs/big_aug_${alg}_${coeff}e${pow}_sorted.txt
        else
            echo "../data/graphs/big_aug_${alg}_${coeff}e${pow}.txt already exists, skipping..."
        fi

        mv ../data/graphs/big_aug_${alg}_${coeff}e${pow}.txt ../data/graphs/big_aug_${alg}_${coeff}e${pow}_backup.txt

        # For iter in 1 to 4
        for ((iter=2; iter<=5; iter++)); do

            if [ ! -f "../data/graphs/big_aug_${alg}_${coeff}e${pow}_v${iter}.txt" ]; then
                bash big_graph_creator.sh $coeff $pow ${alg} 6

                # Delete temporary files
                rm ../data/graphs/big_aug_${alg}_${coeff}e${pow}_bare.txt
                rm ../data/graphs/big_aug_${alg}_${coeff}e${pow}_sorted.txt
                mv ../data/graphs/big_aug_${alg}_${coeff}e${pow}.txt ../data/graphs/big_aug_${alg}_${coeff}e${pow}_v${iter}.txt
            else
                echo "../data/graphs/big_aug_${alg}_${coeff}e${pow}_v${iter}.txt already exists, skipping..."
            fi
        done

        mv ../data/graphs/big_aug_${alg}_${coeff}e${pow}_backup.txt ../data/graphs/big_aug_${alg}_${coeff}e${pow}.txt

    done

    # Create random restricted graphs
    m=$(($coeff * 10**$pow))
    nrand=$(($m / 6))
    p=$(echo "$m / ($nrand * ($nrand - 1))" | bc -l)

    if [ ! -f "../data/graphs/big_rand_${coeff}e${pow}.txt" ]; then
        echo "Creating random restricted graph with $nrand nodes and ${coeff}e${pow} edges, with p = ${p}..."
        ../build/CreateGraph random_restricted_graph4 $nrand $p > ../data/graphs/big_rand_${coeff}e${pow}.txt
    else
        echo "../data/graphs/big_rand_${coeff}e${pow}.txt already exists, skipping..."
    fi

    for ((iter=2; iter<=5; iter++)); do
        if [ ! -f "../data/graphs/big_rand_${coeff}e${pow}_v${iter}.txt" ]; then
            echo "Creating random restricted graph with $nrand nodes and ${coeff}e${pow} edges, with p = ${p}..."
            ../build/CreateGraph random_restricted_graph4 $nrand $p > ../data/graphs/big_rand_${coeff}e${pow}_v${iter}.txt
        else
            echo "../data/graphs/big_rand_${coeff}e${pow}_v${iter}.txt already exists, skipping..."
        fi
    done

done



for sizes in "1 6"; do
    size=( $sizes )
    coeff=${size[0]}
    pow=${size[1]}

    for alg in "bfct" "rd1" "rd2" "dfs" "gor"; do
        if [ ! -f "../data/graphs/big_${alg}_${coeff}e${pow}.txt" ]; then
            bash big_graph_creator_dag.sh $coeff $pow ${alg}

            # # Delete temporary files
            # rm ../data/graphs/big_${alg}_${coeff}e${pow}_sorted.txt
        else
            echo "../data/graphs/big_${alg}_${coeff}e${pow}.txt already exists, skipping..."
        fi
    done
done