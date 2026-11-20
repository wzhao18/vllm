#!/bin/bash
S=$1; O=$2
SKIPCAL=1 SUITE=m8 BUILDS="24:64:1:3 12:32:2:3" bash $S/run_m2.sh $S $O/m8
SKIPCAL=1 SUITE=m9 EXTRA=-DONORM_LDG=1 BUILDS="24:64:1:3 16:64:1:1" bash $S/run_m2.sh $S $O/m9
