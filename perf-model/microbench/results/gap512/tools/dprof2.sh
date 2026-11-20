#!/bin/bash
# mb lane gap512: high-B C512 decode profiles that OOM'd with B32-64 in one process (bench state snapshot 30.9 GiB at B92):
# B80 (rows 320) at agentx contexts alone, then B92 (rows 368) alone with contexts capped (NQ_MAX_KV_TOKENS 12M of 17.5M).
G=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/gap512
GAP_TAG=dec-rel1767-b80 GAP_ROWS=320 bash $G/dprof_v.sh
GAP_TAG=dec-rel1767-b92cap GAP_ROWS=368 GAP_CAP=12000000 bash $G/dprof_v.sh
