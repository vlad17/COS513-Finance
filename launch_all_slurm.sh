#!/bin/bash

set -e

if [[ "$#" -ne 5 && "$#" -ne 6 ]]; then
    echo 'Usage: echo 100 1000 | launch_all_slurm.sh R raw_data_dir models_dir lo hi [notify-email]'
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
    echo 'Optionally sends an email to notify a completion of a stage with the'
    echo 'address of the last argument.'
    exit 1
fi

R="$1"
raw_data_dir=$(readlink -f "$2")
models_dir=$(readlink -f "$3")
lo="$4"
hi="$5"
email="$6"

sample_dir=/scratch/vyf/sample
pre_sample_dir=/scratch/vyf/sample-preprocessed
exp_sample_dir=/n/fs/scratch/vyf/sample-expanded

mkdir -p $exp_sample_dir /tmp/vyf $models_dir

# slurm_header runtime mem program name
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
#SBATCH -J gcf-cluster
#SBATCH -o $4-%j.slurm
# Set working directory:
#SBATCH --workdir=/n/fs/gcf/slurm-out
srun /usr/bin/time -f '%E elapsed, %U user, %S system, %M memory, %x status' $3"
}

clusters=$(cat -)

GCF=/n/fs/gcf
FINANCE=$GCF/COS513-Finance
PYENV=$GCF/bin/activate

to_split=/tmp/vyf/all-days-$lo-$hi.txt
ls -1 $raw_data_dir | grep .export.CSV | cut -c1-8 | sort \
  | sed -n "/$lo/,/$hi/p" > $to_split

echo "************************************************************"
echo "STARTING CLUSTER LEARNING"
echo "K = $clusters N =" $(wc -l < $to_split) "R = $R"
echo "************************************************************"

SCRIPT_DIR=/n/fs/gcf/generated-slurm-scripts

# In our scripts we perform an ugly but time-saving hack.
# I have copied the libc-2.17.so to /n/fs/gcf, which is NFS-accessible.
# (this also requires making a symlink named libc.so.6 pointing to it).
# Then we have to alter LD_LIBRARY_PATH so that we look in the right
# place for it.

for i in $(cat $to_split); do
  name="sample-expand-$i"
  slurm_header "01:00:00" "1G" "/bin/bash -c \"
    mkdir -p $sample_dir $pre_sample_dir
    shuf -n 150 $raw_data_dir/$i.export.CSV > $sample_dir/$i.export.CSV
    export LD_LIBRARY_PATH=/n/fs/gcf:\$LD_LIBRARY_PATH
    source $PYENV
    python $FINANCE/preprocessing.py $sample_dir/$i.export.CSV $pre_sample_dir/$i.csv
    python $FINANCE/expand.py $pre_sample_dir/$i.csv $exp_sample_dir/$i.csv
    rm -rf $sample_dir $pre_sample_dir
  \"" "$name" > $SCRIPT_DIR/$name.slurm
done

for i in $clusters; do
  name="sample-learn-$i"
  slurm_header "05:00:00" "8G" "/bin/bash -c \"
    export LD_LIBRARY_PATH=/n/fs/gcf:\$LD_LIBRARY_PATH
    source $PYENV
    python $FINANCE/clustering.py \\\"$exp_sample_dir/*\\\" $models_dir/$i.model $i 
  \"" $name > $SCRIPT_DIR/$name.slurm
done

# TODO now run full-day pipeline
# TODO make mail

echo
echo "************************************************************"
echo "LAUNCHING SAMPLE-EXPANSION STAGE"
echo "************************************************************"
#echo "LAUNCHING SAMPLE-EXPANSION STAGE" | sendmail $email

for i in $(cat $to_split); do
  sbatch $SCRIPT_DIR/sample-expand-$i.slurm
done

echo "************************************************************"
echo "LAUNCHING MODEL LEARNING"
echo "************************************************************"
#echo "LAUNCHING MODEL LEARNING" | sendmail $email

for i in $clusters; do
  sbatch $SCRIPT_DIR/sample-learn-$i.slurm
done
