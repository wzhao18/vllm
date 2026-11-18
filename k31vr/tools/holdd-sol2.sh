#!/bin/bash
# holdd-sol2.sh: rolling node pool for the K3.1 SOL campaign (user 2026-11-17/18: "continuously queue up some nodes; when
# they are allocated, decide whether to keep them or free them" -- node allocation is a high-priority part of the plan).
# Replaces holdd-sol.sh. Three duties, every 2 min:
#  1. ROLLING REQUESTS: per size class keep a target number of PENDING runner2 holds, adding at most one request per class
#     per class cycle, so allocations arrive staggered instead of in bursts. Account = the one with the higher sshare
#     FairShare (coreai_cserve_dev: batch max 2 h; inferencex: 5 h). Demand boost: while more 4-node tasks wait in
#     queue/pending than running pool holds can start, the 4-node target rises by up to +2.
#  2. KEEP / FREE on arrival: a running pool hold (name ...-k31vr-r2*) that has had no task running and no pending queue
#     task it can take for IDLE_MIN minutes is released (it would otherwise burn GPUs on keepalive). Logged as RELEASE.
#  3. kinit -R when < 3 h left.
# Tune: holds/holdd-sol2.conf (bash: T4= T8= C4= C8= IDLE_MIN=); pause requests: touch holds/holdd-sol.stop.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; H=$W/holds; Q=$W/queue; LOG=$H/holdd-sol.log
IA=coreai_comparch_inferencex; CA=coreai_cserve_dev
log() { echo "$(date "+%F %T") [sol2] $*" >> $LOG; }
fs() { sshare -h -A $1 -u $USER -o FairShare 2>/dev/null | awk 'NF{v=$1} END{print v+0}'; }
best_acct() { awk -v a="$(fs $IA)" -v b="$(fs $CA)" 'BEGIN{print (b>a)?"c":"i"}'; }
npend() { squeue -h -u $USER -t PD -o "%j" 2>/dev/null | grep -cE -- "-k31vr-$1$"; }
nrun() { squeue -h -u $USER -t R -o "%j" 2>/dev/null | grep -cE -- "-k31vr-$1$"; }
sub() {  # sub <class-suffix> <nodes> <acct i|c> <pools>
  local acct t tmin out
  if [ "$3" = c ]; then acct=$CA; t=2:00:00; tmin=1:40:00; else acct=$IA; t=$([ $2 -ge 8 ] && echo 5:00:00 || echo 4:00:00); tmin=$([ $2 -ge 8 ] && echo 4:20:00 || echo 2:30:00); fi
  out=$(cd $H && sbatch -A $acct -p batch-xdr,batch-spx -N $2 --segment=$2 --exclusive -t $t --time-min=$tmin \
    -J $acct-k31vr-$1 --export=ALL,POOLS="$4" runner2.sbatch 2>&1 | grep -i submitted)
  log "request $1 ${2}n $acct $t: $out"; }
need_nodes() { case $1 in h8|e2e*|any) echo 99;; nq4|n4*) echo 4;; n2*) echo 2;; *) echo 1;; esac; }
fits_pending() {  # fits_pending <nodes>: 1 if some pending queue task needs <= nodes
  local t p
  for t in $(ls $Q/pending 2>/dev/null | grep -E '\.sh$'); do
    p=$(echo $t | sed -E 's/^[0-9a-z]+-([a-z0-9]+)-.*/\1/')
    [ "$(need_nodes $p)" -le "$1" ] && { echo 1; return; }
  done; echo 0; }
declare -A IDLE   # jobid -> epoch first seen idle
last_kinit=0; last4=0; last8=0
log "holdd-sol2 start (pid $$)"
while true; do
  now=$(date +%s); T4=2; T8=1; C4=1200; C8=3600; IDLE_MIN=25; [ -f $H/holdd-sol2.conf ] && . $H/holdd-sol2.conf
  if (( now - last_kinit > 600 )); then
    exp=$(klist 2>/dev/null | awk '/krbtgt/ {print $3" "$4; exit}'); left=$(( $(date -d "$exp" +%s 2>/dev/null || echo 0) - now ))
    (( left < 10800 )) && kinit -R >> $LOG 2>&1 && log "kinit -R ok ($((left/60)) min left)"; last_kinit=$now
  fi
  # ---- 2. keep / free running pool holds
  for line in $(squeue -h -u $USER -t R -o "%i|%j|%D" | grep -E -- '-k31vr-r2'); do
    J=${line%%|*}; n=${line##*|}
    busy=$(( $(squeue -h -s -j $J -o %j 2>/dev/null | grep -vcE '^(batch|extern|keepalive|keepalive2|ka3)$') + $(ls $Q/running 2>/dev/null | grep -v '\.log$' | grep -cE "\.$J$") ))
    if (( busy > 0 )) || [ "$(fits_pending $n)" = 1 ]; then unset 'IDLE[$J]'; continue; fi
    [ -z "${IDLE[$J]}" ] && IDLE[$J]=$now
    if (( now - IDLE[$J] >= IDLE_MIN * 60 )); then
      log "RELEASE $J (${n}n): idle ${IDLE_MIN} min, no queued task fits"; scancel $J; unset 'IDLE[$J]'
    fi
  done
  # ---- 1. rolling requests
  if [ ! -f $H/holdd-sol.stop ]; then
    a=$(best_acct)
    p4=$(( $(npend r2n4c) + $(npend r2n4q) + $(npend r2n4b) )); r4=$(( $(nrun r2n4c) + $(nrun r2n4q) + $(nrun r2n4b) ))
    w4=$(ls $Q/pending 2>/dev/null | grep -cE '^[0-9a-z]+-(nq4|n4[a-z0-9]*)-.*\.sh$')
    t4=$T4; (( w4 > r4 )) && t4=$(( T4 + ( w4 - r4 > 2 ? 2 : w4 - r4 ) ))
    if (( now - last4 >= C4 )) && (( p4 < t4 )); then
      sub $([ $a = c ] && echo r2n4c || echo r2n4q) 4 $a 'nq4|n4[a-z0-9]*|n2[a-z0-9]*|nq1b?|nqk-[a-z]+|nqz|nqx|nqy|n1[a-z0-9]*'; last4=$now
    fi
    p8=$(( $(npend r2hd8) + $(npend r2n8c) ))
    if (( now - last8 >= C8 )) && (( p8 < T8 )); then
      sub r2hd8 8 i 'h8|e2e[a-z0-9]*|nq4|n4[a-z0-9]*|n2[a-z0-9]*|nq1b?|nqk-[a-z]+|nqz|nqx|nqy|n1[a-z0-9]*|any'; last8=$now
    fi
  fi
  sleep 120
done
