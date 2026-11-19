#!/bin/bash
# EN GPU unit test inside the K3.1 image (cwd = overlay), full output. Args: pytest args.
set -o pipefail
echo "== overlay $(pwd) $(head -1 SOURCE_REV.ws 2>/dev/null) host $(hostname) $(date +%T)"
python3 -m pytest -q -p no:cacheprovider --noconftest -W ignore "$@" 2>&1 | grep -av "^W1118\|Warning\|warn"
echo "== end $(date +%T)"
