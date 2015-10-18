#!/bin/sh

echo "Usage: pipeline news prices theta perplexity dimension"

news=$1
prices=$2
theta=$3
u=$4
dimension=$5

echo $1 $2 $3

echo "Running TSNE on $news"
 ./bh_tsne $news tsne.out $dimension $theta $perplexity

echo "Running HMM on $prices"
Rscript hmm.R $prices states.out

echo "Running model"
./model_linker tsne.out states.out models.out


