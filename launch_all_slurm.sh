#!/bin/bash

set -e

if [[ "$#" -ne 6 && "$#" -ne 7 && "$#" -ne 8 ]]; then
    echo 'Usage: launch_all_slurm.sh R raw_data_dir models_dir summary_dir lo hi [email] [code_dir]'
    echo
    echo 'NOTE: THIS CAN ONLY BE RUN ONCE AT A TIME PER USER'
    echo
    echo 'NOTE: make sure all parameter directories are mountable by NFS,'
    echo 'e.g., have the prefix /n/ on cycles.'
    echo
    echo 'Uses code_dir to run the python files, if supplied. Defaults to'
    echo '/n/fs/gcf/COS513-Finance.'
    echo 
    echo 'Generates a multiple sets set of up to N slurm scripts with 1 CPU per'
    echo 'task and a max runtime of a day. Uses these to run the random-sample'
    echo 'pipeline.'
    echo
    echo 'Performs an R random sampling for every day in the data dir.'
    echo 'Expects raw-data-dir to contain YYYYMMDD.export.CSV files.'
    echo 'Makes sure dates within the interval [lo, hi] are the only'
    echo 'ones selected.'
    echo
    echo 'Writes out the fully-summarized files to summary_dir in YYYYMMDD.csv'
    echo
    echo "Optionally sends an email when everything's done."
    echo
    echo "Example:"
    echo "./launch_all_slurm.sh 150 ../raw-data-20130401-20151021/ /n/fs/scratch/vyf/models /n/fs/scratch/vyf/summaries 20130601 20130703 \$USER@princeton.edu"
    exit 1
fi

R="$1"
raw_data_dir=$(readlink -f "$2")
models_dir=$(readlink -f "$3")
summary_dir=$(readlink -f "$4")
lo="$5"
hi="$6"
email="$7"
code_dir="$8"

sample_dir=/scratch/$USER/sample
pre_sample_dir=/scratch/$USER/sample-preprocessed
exp_sample_dir=/n/fs/scratch/$USER/sample-expanded # saved accross commands to NFS
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


GCF=/n/fs/gcf
FINANCE=$code_dir
PYENV=$GCF/ionic-env/bin/activate

all_days=/tmp/$USER/all-days-$lo-$hi.txt
ls -1 $raw_data_dir | grep .export.CSV | cut -c1-8 | sort \
  | sed -n "/$lo/,/$hi/p" > $all_days

# infinite gmm hyperparameters
alpha=1
max_components=20000

echo "************************************************************"
echo "STARTING IGMM LEARNING"
echo "IGMM(max=$max_components, alpha=$alpha) N =" $(wc -l < $all_days) "R = $R"
echo "************************************************************"

SCRIPT_DIR=/n/fs/gcf/generated-slurm-scripts

for i in $(cat $all_days); do
  name="sample-expand-$i"
  slurm_header "00:10:00" "1G" "/bin/bash -c \"
    set -e
    mkdir -p $sample_dir $pre_sample_dir
    shuf -n $R $raw_data_dir/$i.export.CSV > $sample_dir/$i.export.CSV
    source $PYENV
    python $FINANCE/preprocessing.py $sample_dir/$i.export.CSV $pre_sample_dir/$i.csv
    python $FINANCE/expand.py $pre_sample_dir/$i.csv $exp_sample_dir/$i.csv
    if [ ! -s $exp_sample_dir/$i.csv ]; then
      echo file $exp_sample_dir/$i.csv empty, dropping
      rm $exp_sample_dir/$i.csv
    fi
    rm -rf $sample_dir/$i.export.CSV $pre_sample_dir/$i.export.CSV
  \"" "$name" > $SCRIPT_DIR/$name.slurm

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

    name="day-summary-$i"
    slurm_header "01:00:00" "10G" "/bin/bash -c \"
      set -e
      mkdir -p $summary_dir $summary_dir/$j
      source $PYENV
      python $FINANCE/summarize.py $exp_dir/$i.csv $summary_dir/$i.csv $models_dir/igmm
\"" "$name" > $SCRIPT_DIR/$name.slurm
done


name="sample-learn"
slurm_header "05:00:00" "16G" "/bin/bash -c \"
  set -e
  source $PYENV
  python $FINANCE/infinite_gmm.py $exp_sample_dir $models_dir/igmm $max_components $alpha
\"" $name > $SCRIPT_DIR/$name.slurm

echo
echo "************************************************************"
echo "LAUNCHING SAMPLE-EXPANSION STAGE"
echo "************************************************************"

sample_expansion=()
for i in $(cat $all_days); do
  sample_expansion+=($(sbatch $SCRIPT_DIR/sample-expand-$i.slurm | cut -f4 -d' '))
done
sample_expansion=$(echo ${sample_expansion[@]} | tr ' ' ':')
echo "SLURM JOBS" $sample_expansion

rm -f $SLURM_OUT/sample-expansion-stage-$lo-$hi
notify_email "  sample-expansion-stage-$lo-$hi" > /tmp/$USER/update-sample-expansion
sbatch --dependency=afterok:$sample_expansion /tmp/$USER/update-sample-expansion
while [ ! -f $SLURM_OUT/sample-expansion-stage-$lo-$hi ]; do 
  sleep 5
done

echo
echo "************************************************************"
echo "LAUNCHING MODEL LEARNING"
echo "************************************************************"

model_learn=()
model_learn+=($(sbatch --dependency=afterok:$sample_expansion $SCRIPT_DIR/sample-learn.slurm | cut -f4 -d' '))
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

full_days_exp=()
for i in $(cat $all_days); do
    full_days_exp+=($(sbatch --dependency=afterany:$model_learn $SCRIPT_DIR/day-expand-$i.slurm | cut -f4 -d' '))
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

full_days_summary=()
for i in $(cat $all_days); do
  full_days_summary+=($(sbatch --dependency=afterany:$full_days_exp $SCRIPT_DIR/day-summary-$i.slurm | cut -f4 -d' '))
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
