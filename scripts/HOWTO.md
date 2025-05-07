In the following we always assume you are in the `scripts/` directory.

# Create big graphs
If you want to create an augmented version of BAD GOR (i.e., AUG GOR) with 5e5 edges, such that 1 parts of them come from the original pathologic instance and 5 parts
are augmented edges, then run:
```
bash big_graph_creator.sh 5 5 gor 6
```

It will generate the file `../data/graphs/big_aug_gor_5e5.txt`.

# Run a quick experiment
Assume you want to run an experiment on the previously generated instance with BCF with the following settings:
- diam_apprx = 1
- k_factor = 40
- cutedges = 5

Then run:
```
../build/Main diam_apprx=1 k_factor=40 cutedges=5 "SSSP time BCF ../data/graphs/big_aug_gor_5e5.txt 0 1"
```

# Create base graphs for subsequent experiments
For the number of edges in [5 * 10<sup>5</sup>, 1 * 10<sup>6</sup>, 2 * 10<sup>6</sup>, 5 * 10<sup>6</sup>, 1 * 10<sup>7</sup>, 2 * 10<sup>7</sup>], generate AUG instances of BFCT, GOR, RD1, RD2, and DFS.
Also create BAD instances of BFCT, GOR, RD1, RD2, and DFS for 1 * 10<sup>6</sup> edges.
```
bash create_base_graphs.sh
```

# Run BCF vs GOR on BAD instances

cat ../data/queries/bad_bcf_vs_gor_queries.txt | parallel -j 8 {}

# Run k_factor experiment
```
cat ../data/queries/kfact_queries.txt | parallel -j 8 {}
```

# Run diam_apprx experiment
```
cat ../data/queries/diam_queries.txt | parallel -j 8 {}
```

# Run rand_label experiment
```
cat ../data/queries/randl_queries.txt | parallel -j 8 {}
```

# Run use_lazy experiment
```
cat ../data/queries/uselazy_queries.txt | parallel -j 8 {}
```

# Run BCF vs GOR on AUG instances
```
cat ../data/queries/norm_bcf_vs_gor_queries.txt | parallel -j 8 {}
```

# Run BCF vs GOR on SHIFTED instances
First create AUG GOR with: 5e5, 1e6, 2e6, 5e6, 1e7, and 2e7 edges. Then create all the necessary graphs with:
```
bash create_shift.sh
```

Run the experiments with:
```
cat ../data/queries/shiftv_queries.txt | parallel -j 8 {}
```

# Run BCF vs GOR on USA instances
```
bash create_restr_usa.sh
```

Run the experiments with:
```
cat ../data/queries/usa_bcf_vs_gor_queries.txt | parallel -j 8 {}
```

# Generate plots and tables

```
python3 auto_plots_tabs.py
```