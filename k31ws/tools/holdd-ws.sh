#!/bin/bash
# holdd-ws.sh: rolling node pool for the k31ws kernel campaign on Rhea (run on the login node: setsid nohup).
# Node allocation is part of the plan: holds are requested ahead of need and kept/freed by evidence.
#  1-node class (hold1): keep T1 holds running+pending (pending at most P1), requested one per C1 s on the account with
#     the best FairShare; partition = whichever of batch / backfill is estimated to start first (backfill is
#     preemptible, fine for microbench), batch preferred when it starts within 1 h.
#  8-node class (hold8, EP32 / DP8 work): keep T8 requests running+pending at all times (one segment of 8 nodes) so the
#     queue wait overlaps the 1-node work; requested on the best account / earliest partition.
#  Keep/free: a running hold whose GPUs saw no wsgpu.sh use for IDLE_MIN minutes (pool/use/<job> mtime, else its start)
#     and with no wsgpu.sh request waiting (pool/wait/*) is released. hold8 uses IDLE8_MIN.
#  New running holds get their container imported once (warm) so the first wsgpu.sh run starts fast.
# Tune: pool/holdd.conf (T1 P1 C1 T8 IDLE_MIN IDLE8_MIN); stop requesting: touch pool/STOP. Log: pool/holdd.log.
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; L=$K/pool/holdd.log
IMG=/lustre/fsw/coreai_comparch_inferencex/weizha/images/kimi-k31-wzhao-nov-11-aarch64.sqsh
ACCTS="coreai_cserve_dev coreai_comparch_inferencex"
log() { echo "$(date '+%F %T') $*" >> $L; }
fs() { sshare -h -A $1 -u $USER -o FairShare 2>/dev/null | awk 'NF{v=$1} END{print v+0}'; }
best_acct() { local a b=-1 ba; for a in $ACCTS; do v=$(fs $a); awk -v v=$v -v b=$b 'BEGIN{exit !(v>b)}' && { b=$v; ba=$a; }; done; echo $ba; }
est() {  # est <acct> <part> <nodes> <time> [time-min]: epoch of estimated start
  local s; s=$(cd $K/pool && sbatch --test-only -A $1 -p $2 -N $3 --segment=$3 --exclusive -t $4 ${5:+--time-min=$5} \
    --wrap "sleep 1" 2>&1 | grep -oE "start at [0-9T:-]+" | awk '{print $3}')
  [ -n "$s" ] && date -d "$s" +%s || echo 9999999999; }
sub() {  # sub <class hold1|hold8> <nodes>
  local a now eb ef part t tmin out
  a=$(best_acct); now=$(date +%s)
  eb=$(est $a batch $2 5:00:00); ef=$(est $a backfill $2 8:00:00 2:00:00)
  if (( eb - now <= 3600 )) || (( eb <= ef )); then part=batch; t=5:00:00; tmin=2:00:00; else part=backfill; t=8:00:00; tmin=2:00:00; fi
  out=$(cd $K/pool && sbatch -A $a -p $part -N $2 --segment=$2 --exclusive -t $t --time-min=$tmin -J $a-k31ws.$1 \
    -o $K/pool/logs/$1-%j.out --wrap "hostname; sleep 30000" 2>&1 | grep -i submitted)
  log "request $1 ${2}n $a $part $t (est batch $(date -d @$eb +%H:%M) backfill $(date -d @$ef +%H:%M)): $out"; }
cnt() { squeue -h -u $USER -t $2 -o %j 2>/dev/null | grep -c -- "-k31ws\.$1$"; }
declare -A WARM
last1=0
log "holdd-ws start pid $$"
while :; do
  T1=6; P1=3; C1=300; T8=1; IDLE_MIN=60; IDLE8_MIN=120
  [ -f $K/pool/holdd.conf ] && . $K/pool/holdd.conf
  now=$(date +%s)
  if [ ! -f $K/pool/STOP ]; then
    r1=$(cnt hold1 R); p1=$(cnt hold1 PD)
    if (( r1 + p1 < T1 )) && (( p1 < P1 )) && (( now - last1 >= C1 )); then sub hold1 1; last1=$now; fi
    r8=$(cnt hold8 R); p8=$(cnt hold8 PD)
    (( r8 + p8 < T8 )) && sub hold8 8
  fi
  waiting=$(ls $K/pool/wait 2>/dev/null | wc -l)
  while IFS='|' read -r J N S NAME; do
    if [ -z "${WARM[$J]}" ]; then
      WARM[$J]=1; log "warm container on $J ($N)"
      ( srun --overlap --jobid $J -N$(scontrol show hostnames $N | wc -l) --ntasks-per-node=1 --container-image=$IMG \
          --container-mounts=/lustre:/lustre --no-container-mount-home true >> $K/pool/logs/warm-$J.log 2>&1 & )
    fi
    lu=$(stat -c %Y $K/pool/use/$J 2>/dev/null || date -d "$S" +%s)
    busy=$(ls $K/pool/locks 2>/dev/null | grep -c "^$J\.")  # lock files exist once used; check live flocks below
    live=0; for f in $K/pool/locks/$J.*; do [ -e "$f" ] || continue; flock -n "$f" true || { live=1; break; }; done
    (( live )) && { touch $K/pool/use/$J; continue; }
    lim=$IDLE_MIN; [[ $NAME == *hold8 ]] && lim=$IDLE8_MIN
    if (( waiting == 0 )) && (( now - lu > lim * 60 )); then
      log "RELEASE $J ($NAME $N): no use for $(( (now - lu) / 60 )) min, nothing waiting"; scancel $J
    fi
  done < <(squeue -h -u $USER -t R -o "%i|%N|%S|%j" | grep -- '-k31ws\.hold')
  sleep 60
done
