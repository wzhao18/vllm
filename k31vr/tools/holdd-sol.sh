#!/bin/bash
# holdd-sol.sh: rolling node pool for the K3.1 SOL effort (user 2026-11-17: "continuously queue up some nodes; when they are
# allocated, decide whether to keep them or free them", so runs never wait for a fresh queue).
# - Keeps TARGET pending holds per size class, topping up at most ONE request per class per CYCLE so arrivals are staggered.
# - Holds are runner2 jobs named ...-k31vr-r2hd<N> (5 h, time-min 4h20): the dsrel dispatcher uses an arriving hold for queued
#   work (AgentX + gsm8k) or releases it within ~1 min when nothing fits ("RELEASE ... no work it can take").
# - On demand: 4-node holds (r2n4b) while 4-node microbench tasks (pool n4x) are pending.
# - Renews kinit (< 3 h left), like holdd21.
# Tune: holds/holdd-sol.conf (bash: T6= T8= T14= CYCLE=); stop requesting: touch holds/holdd-sol.stop.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; H=$W/holds; Q=$W/queue; LOG=$H/holdd-sol.log
ACCT=coreai_comparch_inferencex; P=$ACCT-k31vr
log() { echo "$(date "+%F %T") $*" >> $LOG; }
npend() { squeue -h -u $USER -t PD -o "%j" 2>/dev/null | grep -cE "^$P-$1$"; }
sub() {  # sub <suffix> <nodes> <time> <time-min> [extra env]
  local out; out=$(cd $H && sbatch -A $ACCT -p batch-xdr,batch-spx -N $2 --segment=$2 --exclusive -t $3 --time-min=$4 -J $P-$1 \
    --export=ALL,${5:-POOLS=nq4} runner2.sbatch 2>&1 | tail -1); log "request $1 ${2}n $3: $out"; }
last_kinit=0; last_cycle=0
log "holdd-sol start (pid $$)"
while true; do
  now=$(date +%s); T6=1; T8=2; T14=1; CYCLE=5400; [ -f $H/holdd-sol.conf ] && . $H/holdd-sol.conf
  if (( now - last_kinit > 600 )); then
    exp=$(klist 2>/dev/null | awk '/krbtgt/ {print $3" "$4; exit}'); left=$(( $(date -d "$exp" +%s 2>/dev/null || echo 0) - now ))
    if (( left < 10800 )); then kinit -R >> $LOG 2>&1 && log "kinit -R ok ($((left/60)) min left)"; fi; last_kinit=$now
  fi
  if [ ! -f $H/holdd-sol.stop ]; then
    if (( now - last_cycle >= CYCLE )); then
      for c in "6 $T6" "8 $T8" "14 $T14"; do set -- $c
        p=$(npend r2hd$1); (( p < $2 )) && sub r2hd$1 $1 5:00:00 4:20:00
      done
      last_cycle=$now
    fi
    t4=$(ls $Q/pending 2>/dev/null | grep -cE "^[0-9a-z]+-n4[a-z0-9]*-")
    if (( t4 > 0 )) && (( $(npend r2n4b) + $(squeue -h -u $USER -t R -o "%j" | grep -cE "^$P-r2n4b$") == 0 )); then
      sub r2n4b 4 2:00:00 1:15:00 POOLS=n4x
    fi
  fi
  sleep 120
done
