#!/bin/bash
# int_one.sh <pytest args...> (inside the image via wsgpu.sh, cwd = overlay): one test run with failure details.
echo "== overlay $(pwd) $(head -1 SOURCE_REV.ws 2>/dev/null) gpus $CUDA_VISIBLE_DEVICES"
timeout 900 python3 -m pytest -q -p no:cacheprovider --noconftest -rfE "$@" 2>&1 | grep -vE "Warning|warn|deprecat|^  /|^$" | tail -80
