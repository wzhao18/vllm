#!/bin/bash
# launch_ws.sh <log> <wsgpu args...>: detached wsgpu.sh run on the Rhea login node.
LOG=$1; shift
setsid -f /lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/tools/wsgpu.sh "$@" > "$LOG" 2>&1 < /dev/null
echo "launched -> $LOG"
