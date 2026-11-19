#!/bin/bash
# holdd-ws.sh: rolling node pool for the k31ws kernel campaign (run on the cluster's login node: setsid nohup).
# One script for both clusters (XC 2026-11-18); the cluster is taken from the login host name:
#   rhea   : partitions batch / backfill, holds run "sleep" (--wrap)
#   hecate : partitions batch-xdr, batch-spx / backfill-xdr, backfill-spx (xdr = IB, spx = RoCE: every request names
#            ONE partition, so a multi-node hold never spans fabrics); holds run tools/holdka.sh (idle guard for the
#            site reaper); coreai_cserve_dev caps batch jobs at 2 h on Hecate.
# Node allocation is part of the plan: holds are requested ahead of need and kept/freed by evidence.
#  1-node class (hold1): keep T1 holds running+pending (pending at most P1), requested one per C1 s on the account with
#     the best FairShare; partition = whichever batch / backfill partition is estimated to start first (backfill is
#     preemptible, fine for microbench), batch preferred when it starts within 1 h.
#  8-node class (hold8, EP32 / DP8 work): keep T8 requests running+pending at all times (one segment of 8 nodes) so the
#     queue wait overlaps the 1-node work; requested on the best account / earliest partition.
#  Keep/free: a running hold whose GPUs saw no wsgpu.sh use for IDLE_MIN minutes (pool/use/<job> mtime, else its start)
#     and with no wsgpu.sh request waiting (pool/wait/*) is released. hold8 uses IDLE8_MIN. (Campaign policy: both are
#     1000000, i.e. never.)
#  New running holds get their container imported once (warm) so the first wsgpu.sh run starts fast.
# Tune: pool/holdd.conf (T1 P1 C1 T8 TB IDLE_MIN IDLE8_MIN); stop requesting: touch pool/STOP. Log: pool/holdd.log.
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; L=$K/pool/holdd.log
IMG=/lustre/fsw/coreai_comparch_inferencex/weizha/images/kimi-k31-wzhao-nov-11-aarch64.sqsh
ACCTS="coreai_cserve_dev coreai_comparch_inferencex"
case $(hostname -s) in
  login-hecate*) CL=hecate; PBATCH="batch-xdr batch-spx"; PBACK="backfill-xdr backfill-spx"; JOBCMD=($K/tools/holdka.sh) ;;
  *)             CL=rhea;   PBATCH="batch";               PBACK="backfill";                   JOBCMD=(--wrap "hostname; sleep 30000") ;;
esac
PALLB=$(echo $PBATCH | tr ' ' ,)
log() { echo "$(date '+%F %T') $*" >> $L; }
fs() { sshare -h -A $1 -u $USER -o FairShare 2>/dev/null | awk 'NF{v=$1} END{print v+0}'; }
best_acct() { local a b=-1 ba; for a in $ACCTS; do v=$(fs $a); awk -v v=$v -v b=$b 'BEGIN{exit !(v>b)}' && { b=$v; ba=$a; }; done; echo $ba; }
tlim() {  # tlim <acct> <batch|backfill> -> "limit time-min"
  if [ $2 = backfill ]; then echo "8:00:00 2:00:00"
  elif [ $CL = hecate ] && [ $1 = coreai_cserve_dev ]; then echo "2:00:00 1:40:00"
  else echo "5:00:00 2:00:00"; fi; }
est() {  # est <acct> <part> <nodes> <time> [time-min]: epoch of estimated start
  local s; s=$(cd $K/pool && sbatch --test-only -A $1 -p $2 -N $3 --segment=$3 --exclusive -t $4 ${5:+--time-min=$5} \
    -J $1-k31ws.est --wrap "sleep 1" 2>&1 | grep -oE "start at [0-9T:-]+" | awk '{print $3}')
  [ -n "$s" ] && date -d "$s" +%s || echo 9999999999; }
sub() {  # sub <class hold1|hold8> <nodes> [batch|backfill|auto]
  local a now p e eb=9999999999 ef=9999999999 pb pf part t tmin out
  a=$(best_acct); now=$(date +%s)
  for p in $PBATCH; do read -r t tmin < <(tlim $a batch); e=$(est $a $p $2 $t $tmin); (( e < eb )) && { eb=$e; pb=$p; }; done
  for p in $PBACK; do read -r t tmin < <(tlim $a backfill); e=$(est $a $p $2 $t $tmin); (( e < ef )) && { ef=$e; pf=$p; }; done
  [ -z "$pb" ] && pb=${PBATCH%% *}; [ -z "$pf" ] && pf=${PBACK%% *}
  if [ "$3" = batch ] || { [ "${3:-auto}" = auto ] && { (( eb - now <= 3600 )) || (( eb <= ef )); }; }; then
    part=$pb; read -r t tmin < <(tlim $a batch)
  else part=$pf; read -r t tmin < <(tlim $a backfill); fi
  out=$(cd $K/pool && sbatch -A $a -p $part -N $2 --segment=$2 --exclusive -t $t --time-min=$tmin -J $a-k31ws.$1 \
    -o $K/pool/logs/$1-%j.out "${JOBCMD[@]}" 2>&1 | grep -i submitted)
  log "request $1 ${2}n $a $part $t (est batch $pb $(date -d @$eb +%H:%M) backfill $pf $(date -d @$ef +%H:%M)): $out"; }
cnt() { squeue -h -u $USER -t $2 -o %j 2>/dev/null | grep -c -- "-k31ws\.$1$"; }
cntp() { squeue -h -u $USER -t $2 -p $3 -o %j 2>/dev/null | grep -c -- "-k31ws\.$1$"; }  # cntp <class> <state> <partitions>
declare -A WARM
last1=0; lastk=0
log "holdd-ws start pid $$ cluster $CL"
while :; do
  T1=6; P1=3; C1=300; T8=1; TB=6; IDLE_MIN=1000000; IDLE8_MIN=1000000
  [ -f $K/pool/holdd.conf ] && . $K/pool/holdd.conf
  now=$(date +%s)
  # Hecate: sbatch needs a Kerberos ticket (~6 h, renewable for a week): renew hourly
  [ $CL = hecate ] && (( now - lastk > 3600 )) && { kinit -R >/dev/null 2>&1 || log "kinit -R failed (klist: $(klist 2>&1 | grep -i expires -A1 | tail -1))"; lastk=$now; }
  if [ ! -f $K/pool/STOP ]; then
    r1=$(cnt hold1 R); p1=$(cnt hold1 PD)
    if (( r1 + p1 < T1 )) && (( p1 < P1 )) && (( now - last1 >= C1 )); then sub hold1 1; last1=$now; fi
    # backfill holds are preempted ~30 min after start (FAST 2026-11-18): always keep TB non-preemptible batch
    # 1-node holds running+pending in addition, so the pool has a stable core when the batch queue reaches them.
    rb=$(cntp hold1 R $PALLB); pb=$(cntp hold1 PD $PALLB)
    (( rb + pb < TB )) && sub hold1 1 batch
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
    live=0; for f in $K/pool/locks/$J.*; do [ -e "$f" ] || continue; flock -n "$f" true || { live=1; break; }; done
    (( live )) && { touch $K/pool/use/$J; continue; }
    lim=$IDLE_MIN; [[ $NAME == *hold8 ]] && lim=$IDLE8_MIN
    if (( waiting == 0 )) && (( now - lu > lim * 60 )); then
      log "RELEASE $J ($NAME $N): no use for $(( (now - lu) / 60 )) min, nothing waiting"; scancel $J
    fi
  done < <(squeue -h -u $USER -t R -o "%i|%N|%S|%j" | grep -- '-k31ws\.hold')
  sleep 60
done
