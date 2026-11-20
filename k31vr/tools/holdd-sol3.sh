#!/bin/bash
# holdd-sol3.sh: rolling node pool for the K3.1 SOL campaign (replaces holdd-sol2.sh, 2026-11-18).
# Node allocation is a high-priority part of the plan: queue holds periodically, decide keep/free on arrival.
# Lesson from sol2 (05:08): a fixed 25-min idle release freed an 8n hold minutes before two lanes queued 8n/4n
# work, while re-acquiring took 6+ h. sol3 makes the keep/free decision cost-aware:
#  1. ROLLING REQUESTS per class (4n, 8n): keep T4/T8 pending holds, at most one new request per class per
#     C4/C8 s, on the account with the best FairShare. If the batch estimate is > BF_AFTER_H hours away and a
#     backfill partition starts at least 1 h sooner, the request goes to backfill-{spx,xdr} (preemptible, fine
#     for microbench). Demand boost: +up to 2 on the 4n target while 4n tasks wait beyond running 4n holds.
#  2. KEEP / FREE: a running pool hold (name *-k31vr-r2*) with no task running is released only when ALL hold:
#     - no pending queue task fits it (h8 needs >= 8 nodes, nq4/n4 4, n2 2, else 1),
#     - no live demand marker fits it: holds/demand/<lane> containing "<nodes> <expiry-epoch>" (lanes that are
#       about to queue work write one; expired markers are ignored),
#     - it has been idle >= the adaptive window: IDLE_MIN (25) when a same-size hold would start within 1 h,
#       otherwise IDLE_MAX (90) -- re-acquiring is expensive, so idle tolerance scales with the wait.
#  3. kinit -R when < 3 h left.
# Tune: holds/holdd-sol3.conf (T4 T8 C4 C8 IDLE_MIN IDLE_MAX BF_AFTER_H); pause requests: touch holds/holdd-sol.stop.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; H=$W/holds; Q=$W/queue; LOG=$H/holdd-sol.log; D=$H/demand
IA=coreai_comparch_inferencex; CA=coreai_cserve_dev
mkdir -p $D
log() { echo "$(date "+%F %T") [sol3] $*" >> $LOG; }
fs() { sshare -h -A $1 -u $USER -o FairShare 2>/dev/null | awk 'NF{v=$1} END{print v+0}'; }
best_acct() { awk -v a="$(fs $IA)" -v b="$(fs $CA)" 'BEGIN{print (b>a)?"c":"i"}'; }
npend() { squeue -h -u $USER -t PD -o "%j" 2>/dev/null | grep -cE -- "-k31vr-($1)$"; }
nrun() { squeue -h -u $USER -t R -o "%j" 2>/dev/null | grep -cE -- "-k31vr-($1)$"; }
est_start() {  # est_start <acct> <part> <nodes> <time> -> epoch of estimated start (or far future)
  local s; s=$(cd $H && sbatch --test-only -J $1-k31vr-probe -A $1 -p $2 -N $3 --segment=$3 --exclusive -t $4 \
    runner2.sbatch 2>&1 | grep -oE "start at [0-9T:-]+" | awk '{print $3}')
  [ -n "$s" ] && date -d "$s" +%s || echo 9999999999; }
declare -A WAIT_CACHE WAIT_AT
wait_s() {  # seconds until a new <n>-node batch hold would start (cached 10 min)
  local n=$1 now; now=$(date +%s)
  if [ -z "${WAIT_AT[$n]}" ] || (( now - WAIT_AT[$n] > 600 )); then
    local e; e=$(est_start $IA batch-xdr,batch-spx $n 2:00:00); WAIT_CACHE[$n]=$(( e - now )); WAIT_AT[$n]=$now
  fi; echo ${WAIT_CACHE[$n]}; }
sub() {  # sub <class-suffix> <nodes> <acct i|c> <pools>
  local acct t tmin part out sfx=$1 bstart bfstart bfp now
  if [ "$3" = c ]; then acct=$CA; t=2:00:00; tmin=1:40:00; else acct=$IA; t=$([ $2 -ge 8 ] && echo 5:00:00 || echo 4:00:00); tmin=$([ $2 -ge 8 ] && echo 4:20:00 || echo 2:30:00); fi
  part=batch-xdr,batch-spx; now=$(date +%s)
  bstart=$(est_start $acct $part $2 $t)
  if (( bstart - now > BF_AFTER_H * 3600 )); then
    for bfp in backfill-spx backfill-xdr; do
      bfstart=$(est_start $IA $bfp $2 4:00:00)
      if (( bfstart + 3600 < bstart )); then part=$bfp; acct=$IA; t=4:00:00; tmin=2:00:00; sfx=${1/r2/r2bf}; bstart=$bfstart; fi
    done
  fi
  out=$(cd $H && sbatch -A $acct -p $part -N $2 --segment=$2 --exclusive -t $t --time-min=$tmin \
    -J $acct-k31vr-$sfx --export=ALL,POOLS="$4" runner2.sbatch 2>&1 | grep -i submitted)
  log "request $sfx ${2}n $acct $part $t (est $(date -d @$bstart +%H:%M)): $out"; }
need_nodes() { case $1 in h8|e2e*|any) echo 8;; nq4|n4*) echo 4;; n2*) echo 2;; *) echo 1;; esac; }
fits_pending() {  # 1 if some pending queue task needs <= $1 nodes
  local t p
  for t in $(ls $Q/pending 2>/dev/null | grep -E '\.sh$'); do
    p=$(echo $t | sed -E 's/^[0-9a-z]+-([a-z0-9]+)-.*/\1/')
    [ "$(need_nodes $p)" -le "$1" ] && { echo 1; return; }
  done; echo 0; }
fits_demand() {  # 1 if a live demand marker asks for <= $1 nodes
  local f n e now; now=$(date +%s)
  for f in $D/*; do [ -f "$f" ] || continue; read n e < "$f"
    [ -n "$e" ] && (( e > now )) && (( n <= $1 )) && { echo 1; return; }; done; echo 0; }
declare -A IDLE
last_kinit=0; last4=0; last8=0
log "holdd-sol3 start (pid $$)"
while true; do
  now=$(date +%s); T4=2; T8=1; C4=1200; C8=3600; IDLE_MIN=25; IDLE_MAX=90; BF_AFTER_H=3
  [ -f $H/holdd-sol3.conf ] && . $H/holdd-sol3.conf
  if (( now - last_kinit > 600 )); then
    exp=$(klist 2>/dev/null | awk '/krbtgt/ {print $3" "$4; exit}'); left=$(( $(date -d "$exp" +%s 2>/dev/null || echo 0) - now ))
    (( left < 10800 )) && kinit -R >> $LOG 2>&1 && log "kinit -R ok ($((left/60)) min left)"; last_kinit=$now
  fi
  # ---- 2. keep / free running pool holds
  for line in $(squeue -h -u $USER -t R -o "%i|%j|%D" | grep -E -- '-k31vr-r2'); do
    J=${line%%|*}; n=${line##*|}
    busy=$(( $(squeue -h -s -j $J -o %j 2>/dev/null | grep -vcE '^(batch|extern|keepalive|keepalive2|ka3)$') + $(ls $Q/running 2>/dev/null | grep -v '\.log$' | grep -cE "\.$J$") ))
    if (( busy > 0 )) || [ "$(fits_pending $n)" = 1 ] || [ "$(fits_demand $n)" = 1 ]; then unset 'IDLE[$J]'; continue; fi
    [ -z "${IDLE[$J]}" ] && IDLE[$J]=$now
    ws=$(wait_s $n); win=$IDLE_MIN; (( ws > 3600 )) && win=$IDLE_MAX
    if (( now - IDLE[$J] >= win * 60 )); then
      log "RELEASE $J (${n}n): idle ${win} min (re-acquire est $((ws/60)) min), no queued task or demand fits"; scancel $J; unset 'IDLE[$J]'
    fi
  done
  # ---- 1. rolling requests
  if [ ! -f $H/holdd-sol.stop ]; then
    a=$(best_acct)
    p4=$(npend 'r2n4c|r2n4q|r2n4b|r2bfn4c|r2bfn4q'); r4=$(nrun 'r2n4c|r2n4q|r2n4b|r2bfn4c|r2bfn4q')
    w4=$(ls $Q/pending 2>/dev/null | grep -cE '^[0-9a-z]+-(nq4|n4[a-z0-9]*)-.*\.sh$')
    t4=$T4; (( w4 > r4 )) && t4=$(( T4 + ( w4 - r4 > 2 ? 2 : w4 - r4 ) ))
    if (( now - last4 >= C4 )) && (( p4 < t4 )); then
      sub $([ $a = c ] && echo r2n4c || echo r2n4q) 4 $a 'nq4|n4[a-z0-9]*|n2[a-z0-9]*|nq1b?|nqk-[a-z]+|nqz|nqx|nqy|n1[a-z0-9]*'; last4=$now
    fi
    p8=$(npend 'r2hd8|r2n8c|r2bfhd8'); w8=$(ls $Q/pending 2>/dev/null | grep -cE '^[0-9a-z]+-(h8|e2e[a-z0-9]*)-.*\.sh$')
    t8=$T8; (( w8 > 0 )) && (( $(nrun 'r2hd8|r2n8c|r2bfhd8') == 0 )) && t8=$(( T8 + 1 ))
    if (( now - last8 >= C8 )) && (( p8 < t8 )); then
      sub r2hd8 8 i 'h8|e2e[a-z0-9]*|nq4|n4[a-z0-9]*|n2[a-z0-9]*|nq1b?|nqk-[a-z]+|nqz|nqx|nqy|n1[a-z0-9]*|any'; last8=$now
    fi
  fi
  sleep 120
done
