#!/bin/bash

set -e

if [[ "$#" -ne 6 && "$#" -ne 7 && "$#" -ne 8 ]]; then
    echo 'Usage: echo [K1 K2 K3 ...] | launch_all_slurm.sh sample-file raw_data_dir models_dir summary_dir lo hi [email] [code_dir]'
    echo
    echo 'NOTE: THIS CAN ONLY BE RUN ONCE AT A TIME PER USER'
    echo
    echo 'NOTE: make sure all parameter directories are mountable by NFS,'
    echo 'e.g., have the prefix /n/ on cycles.'
    echo
    echo 'Uses code_dir to run the python files, if supplied. Defaults to'
    echo '/n/fs/gcf/COS513-Finance.'
    echo
    echo 'Reads a list of cluster sizes from stdin. Uses the parameter sample-file'
    echo 'to preprocess, expand, and learn a clustering model based on, which is then'
    echo 'used for the day-summaries pipeline.'
    echo 
    echo 'Generates a multiple sets set of up to N slurm scripts with 1 CPU per'
    echo 'task and a max runtime of a day.'
    echo
    echo 'Expects raw-data-dir to contain YYYYMMDD.export.CSV files.'
    echo 'Makes sure dates within the interval [lo, hi] are the only'
    echo 'ones selected.'
    echo
    echo 'Writes out the fully-summarized files to summary_dir in YYYYMMDD.csv'
    echo
    echo "Optionally sends an email when everything's done."
    echo
    echo "Example:"
    echo "echo 10 100 | ./launch_all_slurm.sh /n/fs/gcf/raw-data-20130401-20151021/20130401.export.CSV /n/fs/gcf/raw-data-20130401-20151021/ /n/fs/scratch/\$USER/models /n/fs/scratch/\$USER/summaries 20130601 20130703 \$USER@princeton.edu \$(pwd)"
    exit 1
fi

# TODO: smarter dependencies (start training model as soon as expanded is done)
# TODO: run training one model at a time rather than all synchronously.

sample_file=$(readlink -f "$1")
raw_data_dir=$(readlink -f "$2")
models_dir=$(readlink -f "$3")
summary_dir=$(readlink -f "$4")
lo="$5"
hi="$6"
email="$7"
code_dir="$8"

sample_dir=/scratch/$USER/sample
pre_sample_dir=/scratch/$USER/sample-preprocessed
exp_sample_dir=/scratch/$USER/sample-expanded
pre_dir=/scratch/$USER/preprocessed
exp_dir=/n/fs/scratch/$USER/expanded

mkdir -p $exp_sample_dir /tmp/$USER $models_dir $summary_dir

# notify_email description
function notify_email() {
slurm_header "00:05:00" "100mb" "echo done" $1 "
#SBATCH --mail-type=end
#SBATCH --mail-user=$email
"
}

# slurm_header runtime mem program name [additional_sbatch_instr]
SLURM_OUT=/n/fs/gcf/slurm-out/$USER
mkdir -p $SLURM_OUT
function slurm_header() {
echo "#!/bin/sh
# Request runtime
#SBATCH --time=$1
# Request a number of CPUs per task:
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
# Request a number of nodes:
#SBATCH --nodes=1
# Request an amount of memory per node:
#SBATCH --mem=$2
# Specify a job name:
#SBATCH -J gcf-cluster-$4
#SBATCH -o $4
# Set working directory:
#SBATCH --workdir=$SLURM_OUT
$5
srun /usr/bin/time -f '%E elapsed, %U user, %S system, %M memory, %x status' $3"
}

clusters=$(cat -)

GCF=/n/fs/gcf
FINANCE=$code_dir
PYENV=$GCF/ionic-env/bin/activate

all_days=/tmp/$USER/all-days-$lo-$hi.txt
ls -1 $raw_data_dir | grep .export.CSV | cut -c1-8 | sort \
  | sed -n "/$lo/,/$hi/p" > $all_days

echo "************************************************************"
echo "STARTING CLUSTER LEARNING"
echo "K = $clusters N =" $(wc -l < $all_days)
echo "Rows in sample =" $(wc -l < $sample_file)
echo "************************************************************"

SCRIPT_DIR=/n/fs/gcf/generated-slurm-scripts

for i in $clusters; do
  name="sample-learn-$i"
  slurm_header "05:00:00" "46G" "/bin/bash -c \"
    set -e
    mkdir -p $pre_sample_dir $exp_sample_dir
    source $PYENV
    python $FINANCE/preprocessing.py $sample_file $pre_sample_dir/sample-pre-$i.csv
    python $FINANCE/expand.py $pre_sample_dir/sample-pre-$i.csv $exp_sample_dir/sample-exp-$i.csv
    python $FINANCE/clustering.py \\\"$exp_sample_dir/sample-exp-$i.csv\\\" $models_dir/$i.model $i
    rm -rf $pre_sample_dir/sample-pre-$i.csv
    rm -rf $exp_sample_dir/sample-exp-$i.csv
  \"" "$name" > $SCRIPT_DIR/$name.slurm
done

for i in $(cat $all_days); do
  name="day-expand-$i"
  slurm_header "00:30:00" "2G" "/bin/bash -c \"
    set -e
    mkdir -p $pre_dir $exp_dir
    source $PYENV
    python $FINANCE/preprocessing.py $raw_data_dir/$i.export.CSV $pre_dir/$i.csv
    cd $FINANCE # TODO ugly dep for models/
    python $FINANCE/expand.py $pre_dir/$i.csv $exp_dir/$i.csv
    if [ ! -s $exp_dir/$i.csv ]; then
      echo file $exp_dir/$i.csv empty, dropping
      rm $exp_dir/$i.csv
    fi
    rm -rf $pre_dir/$i.csv
  \"" "$name" > $SCRIPT_DIR/$name.slurm

  for j in $clusters; do
      name="day-summary-$i-$j"
      slurm_header "01:00:00" "12G" "/bin/bash -c \"
        set -e
        mkdir -p $summary_dir $summary_dir/$j
        source $PYENV
        python $FINANCE/summarize.py $exp_dir/$i.csv $summary_dir/$j/$i.csv $models_dir/$j.model
  \"" "$name" > $SCRIPT_DIR/$name.slurm
  done
done

echo
echo "************************************************************"
echo "LAUNCHING SAMPLE CLUSTER LEARNING STAGE"
echo "************************************************************"

model_learn=()
for i in $clusters; do
  model_learn+=($(sbatch $SCRIPT_DIR/sample-learn-$i.slurm | cut -f4 -d' '))
done
model_learn=$(echo ${model_learn[@]} | tr ' ' ':')
echo "SLURM JOBS" $model_learn

rm -f $SLURM_OUT/sample-learning-stage-$lo-$hi
notify_email "sample-learning-stage-$lo-$hi" > /tmp/$USER/update-sample-learning
sbatch --dependency=afterok:$model_learn /tmp/$USER/update-sample-learning
while [ ! -f $SLURM_OUT/sample-learning-stage-$lo-$hi ]; do
  sleep 5
done

echo
echo "************************************************************"
echo "LAUNCHING FULL-DAY EXPANSIONS"
echo "************************************************************"

#--dependency=afterany:$model_learn 

full_days_exp=()
for i in $(cat $all_days); do
    full_days_exp+=($(sbatch --dependency=afterany:$model_learn  $SCRIPT_DIR/day-expand-$i.slurm | cut -f4 -d' '))
done
full_days_exp=$(echo ${full_days_exp[@]} | tr ' ' ':')
echo "SLURM JOBS" $full_days_exp

rm -f $SLURM_OUT/full-days-expand-stage-$lo-$hi
notify_email "full-days-expand-stage-$lo-$hi" > /tmp/$USER/update-full-days-expand
sbatch --dependency=afterany:$full_days_exp /tmp/$USER/update-full-days-expand
while [ ! -f $SLURM_OUT/full-days-expand-stage-$lo-$hi ]; do 
  sleep 5
done

echo
echo "************************************************************"
echo "LAUNCHING FULL-DAY SUMMARIES"
echo "************************************************************"

#  --dependency=afterany:$full_days_exp

full_days_summary=()
# Note intentional for-loop-order inversion here so we can finish models sequentially.
for j in $clusters; do
    for i in $(cat $all_days); do
    full_days_summary+=($(sbatch --dependency=afterany:$full_days_exp $SCRIPT_DIR/day-summary-$i-$j.slurm | cut -f4 -d' '))
  done
done
full_days_summary=$(echo ${full_days_summary[@]} | tr ' ' ':')
echo "SLURM JOBS" $full_days_summary

rm -f $SLURM_OUT/full-days-summary-stage-$lo-$hi
notify_email "full-days-summary-stage-$lo-$hi" > /tmp/$USER/update-full-days-summary
sbatch --dependency=afterany:$full_days_summary /tmp/$USER/update-full-days-summary
while [ ! -f $SLURM_OUT/full-days-summary-stage-$lo-$hi ]; do 
  sleep 5
done

echo
echo "************************************************************"
echo "DONE WITH EVERYTHING"
echo "************************************************************"

# TODO cleanup process for expansion files (sample and normal).
# Keep the models around.
