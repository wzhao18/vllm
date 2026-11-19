#!/bin/bash
# D-ATTN: one test file, failures only. $1 = overlay, $2 = test path
cd $1
python3 -m pytest --noconftest -x -q -p no:warnings --tb=short $2 2>&1 | grep -v "^  /\|Deprecat\|Ignoring" | tail -60
