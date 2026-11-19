#!/bin/bash
# Opcode histogram of the kept PTX of the score kernel + the instructions inside the innermost
# loop that contains the BF16 FMAs (between its label and back-branch). Usage: ptx_hist.sh <dump_dir>
D=$1
for f in $(find $D -name "*IndexerScore*.ptx"); do
  echo "== $(basename $f | cut -c1-60)"
  grep -E "^\s+[a-z@]" $f | sed 's/^\s*//; s/^@%p[0-9]* //; s/^@!%p[0-9]* //' | awk '{print $1}' | sed 's/\..*//' | sort | uniq -c | sort -rn | head -40 | tr '\n' ' '; echo
  python3 - "$f" <<'PY'
import re, sys
lines = open(sys.argv[1]).read().split('\n')
# find the loop body containing fma.rn.bf16x2: nearest preceding label and following bra to it
idx = [i for i, l in enumerate(lines) if 'fma.rn.bf16x2' in l]
if idx:
    i0 = idx[0]
    lab = None
    for j in range(i0, -1, -1):
        m = re.match(r'^(\$L\w+):', lines[j].strip())
        if m:
            lab = m.group(1); start = j; break
    # find back-branch to lab after
    end = None
    for j in range(i0, len(lines)):
        if ('bra' in lines[j]) and lab and (lab + ';') in lines[j] and j > i0:
            end = j
    # use outermost loop containing the fma: search labels with a later bra back
    labels = [(j, re.match(r'^(\$L\w+):', lines[j].strip()).group(1)) for j in range(0, i0) if re.match(r'^\$L\w+:', lines[j].strip())]
    best = None
    for j, l in labels[::-1]:
        for k in range(i0, len(lines)):
            if 'bra' in lines[k] and (l + ';') in lines[k]:
                best = (j, k, l); break
        if best: break
    if best:
        j, k, l = best
        body = [x.strip() for x in lines[j:k+1] if re.match(r'^\s+[a-z@]', x)]
        ops = {}
        for x in body:
            x = re.sub(r'^@!?%p\d+ ', '', x)
            op = x.split()[0]
            ops[op] = ops.get(op, 0) + 1
        print(f"loop {l} lines {j}-{k}: {len(body)} instrs:", sorted(ops.items(), key=lambda t: -t[1])[:45])
PY
done
