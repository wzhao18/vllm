#!/bin/bash
# fe_mon.sh <outdir>: every 20 s, per-thread CPU jiffies (utime+stime) of every Dynamo frontend / nginx process on this host.
# Output: <outdir>/femon-<host>.tsv: epoch host pid tid comm jiffies. Host-side (no container), read-only /proc access.
O=${1:?outdir}; h=$(hostname -s); f=$O/femon-$h.tsv; mkdir -p $O
while :; do
  t=$(date +%s.%N)
  for p in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null) || continue
    case "$c" in *dynamo.frontend*|*"nginx: worker"*|*"nginx: master"*) ;; *) continue;; esac
    pid=${p#/proc/}
    for tk in $p/task/*; do
      read -r st < $tk/stat 2>/dev/null || continue
      comm=${st#*(}; comm=${comm%)*}; rest=${st##*) }
      set -- $rest
      echo -e "$t\t$h\t$pid\t${tk##*/}\t${comm// /_}\t$(( ${12} + ${13} ))"
    done
  done >> $f
  sleep 20
done
