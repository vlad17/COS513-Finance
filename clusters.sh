#!/bin/bash

set -e

if [[ "$#" -ne 4 || !(-d "$1") ]]; then
    echo 'Usage: clusters.sh raw-data-dir models-dir low hi'
    echo
    echo 'Computes cluster models and writes them to the models-dir.'
    echo 'Expects raw-data-dir to contain YYYYMMDD.export.CSV files.'
    echo 'Makes sure dates within the interval [low, hi] are the only'
         'ones selected.'
    exit 1
fi

raw_data_dir=$(readlink -f "$1")
sample_dir=/n/fs/scratch/vyf/sample
pre_sample_dir=/n/fs/scratch/vyf/sample-preprocessed
exp_sample_dir=/n/fs/scratch/vyf/sample-expanded
models_dir=$(readlink -f "$2")
low="$3"
hi="$4"

gcf=/n/fs/gcf
PYTHON=$gcf/bin/python
FINANCE=$gcf/COS513-Finance

mkdir -p $models_dir $sample_dir $pre_sample_dir $exp_sample_dir

ls -1 $raw_data_dir | grep .export.CSV | cut -c1-8 | sort \
  | sed -n "/$low/,/$hi/p" \
  | xargs --max-procs=$(nproc) --replace --verbose /bin/bash -c "
shuf -n 150 $raw_data_dir/{}.export.CSV > $sample_dir/{}.export.CSV
$PYTHON $FINANCE/preprocessing.py $sample_dir/{}.export.CSV $pre_sample_dir/{}.csv
$PYTHON $FINANCE/expand.py $pre_sample_dir/{}.csv $exp_sample_dir/{}.csv
"
echo '***********************************************************'
echo 'DONE WITH SAMPLING, PREPROCESSING, EXPANSION. CLUSTERING...'
echo '***********************************************************'

echo $(seq 10 10 50) $(seq 100 100 500) $(seq 1000 1000 5000) | tr ' ' '\n' \
  | xargs --max-procs=$(nproc) --replace --verbose /bin/bash -c "
$PYTHON $FINANCE/clustering.py \"$exp_sample_dir/*\" $models_dir/{}.model {}
"
