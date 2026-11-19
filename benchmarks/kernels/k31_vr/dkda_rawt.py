# D-KDA helper: transpose an ncu --page raw --csv dump (header row + one kernel row) to name=value lines.
import csv
import sys

rows = [r for r in csv.reader(sys.stdin) if r]
hdr = next(r for r in rows if "ID" in r[:3] or "Kernel Name" in r)
vals = rows[-1]
for h, v in zip(hdr, vals):
    if "__" in h or "duration" in h:
        print(f"{h} = {v}")
