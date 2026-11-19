"""Print the largest numeric metrics of an `ncu --page raw --csv` export (last row)."""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
if len(rows) < 3:
    sys.exit()
out = []
for k, v in zip(rows[0], rows[-1]):
    try:
        out.append((float(v.replace(",", "")), k))
    except ValueError:
        pass
for v, k in sorted(out, reverse=True)[: int(sys.argv[2]) if len(sys.argv) > 2 else 30]:
    print(f"{k[:90]:90s} {v:12.2f}")
