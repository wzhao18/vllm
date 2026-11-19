#!/bin/bash
# EN GPU unit test inside the K3.1 image (cwd = overlay). Args: test file(s) + pytest args.
set -o pipefail
echo "== overlay $(pwd) $(head -1 SOURCE_REV.ws 2>/dev/null) host $(hostname) $(date +%T)"
python3 -m pytest -q -p no:cacheprovider --noconftest -rA --durations=5 "$@" 2>&1 | tail -60
echo "== end $(date +%T)"
