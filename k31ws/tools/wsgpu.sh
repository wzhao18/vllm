#!/bin/bash
# wsgpu.sh [-g 1|2|4|8n] [-m MINUTES] [-w WAITMIN] [-o OVTAG] [-i IMG] [-n NAME] -- CMD [ARGS...]
# Run CMD inside the K3.1 image on free GPUs of the shared k31ws node pool on Rhea (run it on the Rhea login node).
#  -g   GPUs: 1, 2 or 4 on one node (default 1), or 8n = a whole 8-node pool hold (EP32/DP8 work; CMD then runs ONCE
#       on the login node with WS_JOB / WS_NODES exported so it can srun --overlap itself across the 8 nodes).
#  -m   expected runtime in minutes (default 60): only holds with at least m+5 minutes left are used.
#  -w   how long to wait for free GPUs (default 240 min) before giving up (exit 75).
#  -o   overlay tag under $K/ov (default base-4fbaa15); the image's vllm is replaced by $K/ov/<tag> on PYTHONPATH.
#  -n   label for the usage log (use your workstream id, e.g. D-TOPK).
# Inside the container, wsenv.sh sets PYTHONPATH, Lustre caches, CUDA_VISIBLE_DEVICES=<the locked GPUs>, then execs CMD.
# GPUs are locked per (hold, gpu) with flock on Lustre for the whole run: two runs never share a GPU.
# Pool holds are the jobs named *-k31ws.hold* (maintained by holdd-ws.sh). Never scancel them yourself.
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
G=1; MIN=60; WAIT=240; OV=base-4fbaa15; NAME=${WS_NAME:-anon}
IMG=/lustre/fsw/coreai_comparch_inferencex/weizha/images/kimi-k31-wzhao-nov-11-aarch64.sqsh
while getopts g:m:w:o:i:n: o; do
  case $o in g) G=$OPTARG;; m) MIN=$OPTARG;; w) WAIT=$OPTARG;; o) OV=$OPTARG;; i) IMG=$OPTARG;; n) NAME=$OPTARG;;
    *) exit 2;; esac
done
shift $((OPTIND-1)); [ "$1" = -- ] && shift
[ $# -gt 0 ] || { echo "usage: wsgpu.sh [-g 1|2|4|8n] [-m min] [-w waitmin] [-o ovtag] [-n name] -- cmd ..."; exit 2; }
case $G in 1|2|4|8n) ;; *) echo "-g must be 1, 2, 4 or 8n"; exit 2;; esac
[ -d $K/ov/$OV ] || { echo "no overlay $K/ov/$OV (stage it with mkov.sh)"; exit 2; }
mkdir -p $K/pool/locks $K/pool/use $K/pool/wait
WF=$K/pool/wait/$(hostname -s).$$; echo "$NAME $G $(date +%s)" > $WF
trap 'rm -f $WF' EXIT
T0=$(date +%s)
try_lock() {  # try_lock <lockfile...>: open + flock -n each, keep fds; on failure release all
  LOCKFDS=()
  local f fd
  for f in "$@"; do
    exec {fd}>"$f"
    if flock -n $fd; then LOCKFDS+=($fd); else exec {fd}>&-; for fd in "${LOCKFDS[@]}"; do exec {fd}>&-; done; LOCKFDS=(); return 1; fi
  done; return 0; }
while :; do
  now=$(date +%s)
  if [ "$G" = 8n ]; then
    while IFS='|' read -r J N E; do
      left=$(( $(date -d "$E" +%s 2>/dev/null || echo 0) - now ))
      (( left > (MIN + 5) * 60 )) || continue
      L=(); for g in 0 1 2 3; do for h in $(scontrol show hostnames $N); do L+=($K/pool/locks/$J.$h.$g); done; done
      if try_lock "${L[@]}"; then
        rm -f $WF; touch $K/pool/use/$J
        echo "$(date +%FT%T) $NAME job=$J nodes=$N gpus=all-32 ov=$OV start" >> $K/pool/usage.log
        echo "wsgpu: $NAME on 8-node hold $J ($N)"
        WS_JOB=$J WS_NODES=$N WS_OV=$OV WS_IMG=$IMG "$@"; rc=$?
        touch $K/pool/use/$J
        echo "$(date +%FT%T) $NAME job=$J rc=$rc end" >> $K/pool/usage.log; exit $rc
      fi
    done < <(squeue -h -u $USER -t R -o "%i|%N|%e|%j|%D" | awk -F'|' '$4 ~ /-k31ws\.hold8/ && $5 == 8 {print $1"|"$2"|"$3}')
  else
    while IFS='|' read -r J N E; do
      left=$(( $(date -d "$E" +%s 2>/dev/null || echo 0) - now ))
      (( left > (MIN + 5) * 60 )) || continue
      for h in $(scontrol show hostnames $N); do
        for s in $(seq 0 $G 3); do
          L=(); GL=""; for ((g=s; g<s+G; g++)); do L+=($K/pool/locks/$J.$h.$g); GL="$GL${GL:+,}$g"; done
          if try_lock "${L[@]}"; then
            rm -f $WF; touch $K/pool/use/$J
            echo "$(date +%FT%T) $NAME job=$J node=$h gpus=$GL ov=$OV start" >> $K/pool/usage.log
            echo "wsgpu: $NAME on $h gpus $GL (hold $J)"
            WS_OV=$OV WS_GPUS=$GL srun --overlap --jobid $J -w $h -N1 -n1 --container-image=$IMG \
              --container-mounts=/lustre:/lustre --no-container-mount-home --container-workdir=/tmp \
              bash $K/tools/wsenv.sh "$@"
            rc=$?; touch $K/pool/use/$J
            echo "$(date +%FT%T) $NAME job=$J node=$h gpus=$GL rc=$rc end" >> $K/pool/usage.log; exit $rc
          fi
        done
      done
    # 1-node holds always; 8-node holds too while no 8n request is waiting (their GPUs are idle until EP32 work runs;
    # an 8n request then waits for these runs to end, so keep -m honest)
    H8=1; grep -lq ' 8n ' $K/pool/wait/* 2>/dev/null && H8=0
    done < <(squeue -h -u $USER -t R -o "%i|%N|%e|%j" | awk -F'|' -v h8=$H8 '$4 ~ /-k31ws\.hold1/ || (h8 && $4 ~ /-k31ws\.hold8/) {print $1"|"$2"|"$3}')
  fi
  (( $(date +%s) - T0 > WAIT * 60 )) && { echo "wsgpu: no free ${G} GPU slot within $WAIT min"; exit 75; }
  sleep 30
done
