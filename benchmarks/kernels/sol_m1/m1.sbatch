#!/bin/bash
#SBATCH -J coreai_cserve_dev-sol16.BCM1
#SBATCH -A coreai_cserve_dev
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --exclude=rhea0057
#SBATCH -o /lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/sol16/M1BC/sb-%j.out
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
bash $K/sol16/M1BC/src/m1_run.sh $K/sol16/M1BC/r-$SLURM_JOB_ID ${M1W:-abkcd} ${M1REPS:-3}
