#!/bin/bash
# wsgpu.sh [-g 1|2|4|8n] [-m MINUTES] [-w WAITMIN] [-o OVTAG] [-i IMG] [-n NAME] -- CMD [ARGS...]
# Run CMD inside the K3.1 image on free GPUs of the shared k31ws node pool on Rhea (run it on the Rhea login node).
#  -g   GPUs: 1, 2 or 4 on one node (default 1), or 8n = a whole 8-node pool hold (EP32/DP8 work; CMD then runs ONCE
#       on the login node with WS_JOB / WS_NODES exported so it can srun --overlap itself across the 8 nodes).
#  -m   expected runtime in minutes (default 30): only holds with at least m+5 minutes left are used. Keep it honest
#       (a step bench is ~5 min): a large -m excludes holds near their end.
#  -w   how long to wait for free GPUs (default 240 min) before giving up (exit 75).
#  -o   overlay tag under $K/ov (default base-4fbaa15); the image's vllm is replaced by $K/ov/<tag> on PYTHONPATH.
#  -n   label for the usage log (use your workstream id, e.g. D-TOPK).
# Inside the container, wsenv.sh sets PYTHONPATH, Lustre caches, CUDA_VISIBLE_DEVICES=<the locked GPUs>, then execs CMD.
# GPUs are locked per (hold, gpu) with flock on Lustre for the whole run: two runs never share a GPU.
# Pool holds are the jobs named *-k31ws.hold* (maintained by holdd-ws.sh). Never scancel them yourself.
# Scheduling (FAST 2026-11-18, tools/wsplan.py): fair round-robin across agents, best-fit packing of 1/2-GPU runs onto
# busy nodes, node reservation for whole-node requests (at most nodes-1 while small requests wait), 5 s polling.
# Preemption: backfill holds are preempted (PreemptMode CANCEL); a run killed by its hold's preemption is re-queued
# automatically with its original queue position (at most WS_RETRY=2 times). See the queue: tools/wsq.sh.
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
G=1; MIN=30; WAIT=240; OV=base-4fbaa15; NAME=${WS_NAME:-anon}
IMG=/lustre/fsw/coreai_comparch_inferencex/weizha/images/kimi-k31-wzhao-nov-11-aarch64.sqsh
while getopts g:m:w:o:i:n: o; do
  case $o in g) G=$OPTARG;; m) MIN=$OPTARG;; w) WAIT=$OPTARG;; o) OV=$OPTARG;; i) IMG=$OPTARG;; n) NAME=$OPTARG;;
    *) exit 2;; esac
done
shift $((OPTIND-1)); [ "$1" = -- ] && shift
[ $# -gt 0 ] || { echo "usage: wsgpu.sh [-g 1|2|4|8n] [-m min] [-w waitmin] [-o ovtag] [-n name] -- cmd ..."; exit 2; }
case $G in 1|2|4|8n) ;; *) echo "-g must be 1, 2, 4 or 8n"; exit 2;; esac
[[ $MIN =~ ^[0-9]+$ ]] || { echo "-m must be minutes"; exit 2; }
[ -d $K/ov/$OV ] || { echo "no overlay $K/ov/$OV (stage it with mkov.sh)"; exit 2; }
mkdir -p $K/pool/locks $K/pool/use $K/pool/wait $K/pool/resv $K/pool/run
ME=$(hostname -s).$$
WF=$K/pool/wait/$ME; RF=$K/pool/run/$ME; RESV=''
T0=$(date +%s); echo "$NAME $G $T0 $MIN" > $WF
trap 'rm -f $WF $RF $RESV' EXIT
POLL=${WS_POLL:-5}; RETRY=${WS_RETRY:-2}
ulog() { echo "$(date +%FT%T) $*" >> $K/pool/usage.log; }
try_lock() {  # try_lock <lockfile...>: open + flock -n each, keep fds; on failure release all
  LOCKFDS=()
  local f fd
  for f in "$@"; do
    exec {fd}>"$f"
    if flock -n $fd; then LOCKFDS+=($fd); else exec {fd}>&-; for fd in "${LOCKFDS[@]}"; do exec {fd}>&-; done; LOCKFDS=(); return 1; fi
  done; return 0; }
unlock_all() { local fd; for fd in "${LOCKFDS[@]}"; do exec {fd}>&-; done; LOCKFDS=(); }
preempted() {  # preempted <job> <planned end epoch>: the hold is gone or now ends early (slurm preemption)
  local st e
  st=$(squeue -h -j $1 -o %T 2>/dev/null)
  [ "$st" = RUNNING ] || return 0
  e=$(date -d "$(squeue -h -j $1 -o %e 2>/dev/null)" +%s 2>/dev/null || echo 0)
  (( e < $2 - 60 )); }
tries=0
while :; do
  now=$(date +%s)
  if [ "$G" = 8n ]; then
    while IFS='|' read -r J N E; do
      left=$(( $(date -d "$E" +%s 2>/dev/null || echo 0) - now ))
      (( left > (MIN + 5) * 60 )) || continue
      L=(); for g in 0 1 2 3; do for h in $(scontrol show hostnames $N); do L+=($K/pool/locks/$J.$h.$g); done; done
      if try_lock "${L[@]}"; then
        rm -f $WF; echo "$NAME 8n $J" > $RF; touch $K/pool/use/$J; TS=$(date +%s)
        ulog "$NAME job=$J nodes=$N gpus=all-32 ov=$OV start waited=$((TS - T0))s"
        echo "wsgpu: $NAME on 8-node hold $J ($N) after $((TS - T0)) s"
        WS_JOB=$J WS_NODES=$N WS_OV=$OV WS_IMG=$IMG "$@"; rc=$?
        touch $K/pool/use/$J; rm -f $RF; unlock_all
        ulog "$NAME job=$J rc=$rc end dur=$(( $(date +%s) - TS ))s"
        if [ $rc != 0 ] && (( tries < RETRY )) && preempted $J $(date -d "$E" +%s); then
          tries=$((tries + 1)); ulog "$NAME job=$J PREEMPTED: re-queued (try $tries)"; echo "$NAME $G $T0 $MIN" > $WF; continue 2
        fi
        exit $rc
      fi
    done < <(squeue -h -u $USER -t R -o "%i|%N|%e|%j|%D" | awk -F'|' '$4 ~ /-k31ws\.hold8/ && $5 == 8 {print $1"|"$2"|"$3}')
  else
    read -r A J H GL HEND _ < <(python3 $K/tools/wsplan.py me $WF 2>/dev/null)
    if [ "$A" = RESERVE ] && [ "$K/pool/resv/$J.$H" != "$RESV" ]; then
      rm -f $RESV; RESV=$K/pool/resv/$J.$H; echo $WF > $RESV.$$; mv -f $RESV.$$ $RESV
      ulog "$NAME reserve $J.$H (waited $(( now - T0 ))s)"; echo "wsgpu: $NAME reserved $H (hold $J)"
    elif [ "$A" = ASSIGN ]; then
      L=(); for g in ${GL//,/ }; do L+=($K/pool/locks/$J.$H.$g); done
      if try_lock "${L[@]}"; then
        rm -f $WF $RESV; RESV=''; echo "$NAME $G $J $H $GL" > $RF; touch $K/pool/use/$J; TS=$(date +%s)
        ulog "$NAME job=$J node=$H gpus=$GL ov=$OV start waited=$((TS - T0))s"
        echo "wsgpu: $NAME on $H gpus $GL (hold $J) after $((TS - T0)) s"
        WS_OV=$OV WS_GPUS=$GL srun --overlap --jobid $J -w $H -N1 -n1 --container-image=$IMG \
          --container-mounts=/lustre:/lustre --no-container-mount-home --container-workdir=/tmp \
          bash $K/tools/wsenv.sh "$@"
        rc=$?; touch $K/pool/use/$J; rm -f $RF; unlock_all
        ulog "$NAME job=$J node=$H gpus=$GL rc=$rc end dur=$(( $(date +%s) - TS ))s"
        if [ $rc != 0 ] && (( tries < RETRY )) && preempted $J $HEND; then
          tries=$((tries + 1)); ulog "$NAME job=$J node=$H PREEMPTED: re-queued (try $tries)"
          echo "wsgpu: hold $J preempted during the run: re-queued (try $tries)"; echo "$NAME $G $T0 $MIN" > $WF; continue
        fi
        exit $rc
      fi
    fi
  fi
  (( $(date +%s) - T0 > WAIT * 60 )) && { echo "wsgpu: no free ${G} GPU slot within $WAIT min"; exit 75; }
  sleep $POLL
done
