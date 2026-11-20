#!/bin/bash
# dattn_skel_ncu.sh <outdir> <prebuilt skel dir>: ncu stall/memory sections for one skeleton launch (R384 CL1 W8 L4)
O=$1; B=$2; mkdir -p $O
echo "== host $(hostname) $(date)"; which ncu
ncu --launch-skip 1 --launch-count 1 --section SpeedOfLight --section WarpStateStats --section SchedulerStats \
    --section MemoryWorkloadAnalysis --section LaunchStats --section Occupancy \
    $B/skel 96 1 8 4 0.8 1 448 576 0 0 296 1 22 > $O/ncu.txt 2>&1
grep -vE "^\s*$|----" $O/ncu.txt | grep -E "Duration|Throughput|Stall|Warp Cycles|Issued|Eligible|Active Warps|Registers|Shared Memory|L2 Hit|DRAM|Block Limit|Waves|SM Frequency|Elapsed|One or More|No Eligible" | head -45
echo "== END $(date)"
