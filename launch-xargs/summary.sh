#!/bin/bash

set -e

if [[ "$#" -ne 3 || !(-d "$1") || !(-d "$2") ]]; then
    echo 'Usage: summary.sh raw-data-dir models-dir out-dir'
    echo
    echo 'Computes day-feature summaries'
    echo 'Expects raw-data-dir to contain YYYYMMDD.export.CSV files.'
    echo 'Expects models-dir to contain pickled python *.model files'
    exit 1
fi

raw_data_dir=$(readlink -f "$1")
models_dir=$(readlink -f "$2")
pre_dir=/n/fs/vyf/scratch/preprocess
expand_dir=/n/fs/vyf/scratch/expand
out_dir=$(readlink -f "$3")

gcf=/n/fs/gcf
PYTHON=$gcf/bin/python
FINANCE=$gcf/COS513-Finance

mkdir -p $pre_dir $expand_dir $out_dir

# Note normal $ refer to replacements in this shell, \$ in the child.
ls -1 "$raw_data_dir" | grep ".export.CSV" | cut -c1-8 \
  | xargs --max-procs=$(nproc) --replace --verbose /bin/bash -c "
$PYTHON $FINANCE/preprocessing.py $raw_data_dir/{}.export.CSV $pre_dir/{}.csv
$PYTHON $FINANCE/expand.py $pre_dir/{}.csv $expand_dir/{}.csv
for i in $models_dir/*.model; do
  model_num=\$(basename \$i | cut -f1 -d'.')
  mkdir -p $out_dir/\$model_num
  $PYTHON $FINANCE/summarize.py $expand_dir/{}.csv $out_dir/\$model_num/{}.csv \$i
done
"

