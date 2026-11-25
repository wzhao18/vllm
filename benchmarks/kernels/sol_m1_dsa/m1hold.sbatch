#!/bin/bash
#SBATCH -J coreai_cserve_dev-sol16n1.m1dsa
#SBATCH -A coreai_cserve_dev
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH --exclude=rhea0057
#SBATCH -o /lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/sol16/M1DSA/hold-%j.out
# M1 DSA probe hold (own 1-node hold: no free pool lease). Steps run in it via sol16-n1run.sh; released by scancel of
# this exact job id when the probes end.
echo "hold $SLURM_JOB_ID $(hostname) $(date +%T)"
sleep 5400
