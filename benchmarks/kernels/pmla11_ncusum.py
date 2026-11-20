# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA-2CTA round 3: compact ncu raw-csv summary (one kernel per file).
Usage: python3 pmla11_ncusum.py <regex> <raw.csv> [<raw.csv> ...]
Prints, per file, every raw metric whose name matches <regex> (name, unit, value)."""

import csv
import re
import sys


def main():
    pat = re.compile(sys.argv[1])
    for path in sys.argv[2:]:
        rows = list(csv.reader(open(path)))
        if len(rows) < 3:
            print(path, "empty")
            continue
        names, units, vals = rows[0], rows[1], rows[-1]
        print("==", path.rsplit("/", 1)[-1])
        for n, u, v in zip(names, units, vals):
            if pat.search(n):
                print(f"  {n} [{u}] = {v}")


if __name__ == "__main__":
    main()
