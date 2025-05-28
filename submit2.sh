#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o /u/home/s/skikuchi/scratch/Kaggle/hms/log/train2.$JOB_ID
#$ -j y
# Edit the line below to request the appropriate runtime and memory
# (or to add any other resource) as needed:
#$ -l h_rt=6:00:00,h_data=48G,gpu,A100
# Add multiple cores/nodes as needed:
#$ -pe shared 1

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

cd /u/home/s/skikuchi/scratch/Kaggle/hms
# load the job environment:
. /u/local/Modules/default/init/modules.sh
module load cuda/11.8
module load gcc/11.3.0
module load anaconda3
conda activate /u/home/s/skikuchi/project-xyang123/miniconda3/envs/cellplm

python train2.py

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
### anaconda_python_submit.sh STOP ###