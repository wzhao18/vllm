"""nq_junit.py <gpu stage dir>...: the GPU correctness stage's summary (nq_gputest.sh), from each dir's status.txt
(one 'NQTEST <kind> <file> rc=<rc> secs=<s>' line per pytest / torchrun run) and the pytest junit XML.

Per file: passed / failed / errors / skipped counts, the failing test ids, the skip reasons (a test that skipped did
not run: the coverage is what passed), and flags: FAIL (failures or errors, or a non-zero torchrun / pytest rc),
TIMEOUT (rc 124), NO-TESTS (pytest rc 5), ALL-SKIPPED (every collected test skipped), NO-JUNIT (pytest died before
writing it). A dir with no status line at all is NOT RUN (the stage was killed or never started).
"""
import collections
import os
import re
import sys
import xml.etree.ElementTree as ET

STATUS = re.compile(r"^NQTEST (pytest|torchrun) (\S+) rc=(\S+)(?: np=\d+)? secs=(\d+)")
PLAN = re.compile(r"^NQTEST plan (pytest|torchrun) (\S+)$")
STAGE = re.compile(r"^NQTEST stage (\S+) rc=(\S+)$")


def junit(path):
    counts = collections.Counter()
    failing, skips = [], collections.Counter()
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        kind = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                kind = "failed" if child.tag == "failure" else "errors"
                failing.append(f"{name}: {(child.get('message') or '').splitlines()[0][:160] if child.get('message') else child.tag}")
                break
            if child.tag == "skipped":
                kind = "skipped"
                skips[(child.get("message") or "?")[:120]] += 1
                break
        counts[kind] += 1
    return counts, failing, skips


def stage(d):
    out, bad = [], []
    status = os.path.join(d, "status.txt")
    lines = open(status).read().splitlines() if os.path.isfile(status) else []
    runs = [m.groups() for m in map(STATUS.match, lines) if m]
    none = [line for line in lines if " rc=NONE " in line]
    planned = [m.groups() for m in map(PLAN.match, lines) if m]
    stages = [m.groups() for m in map(STAGE.match, lines) if m]
    out.append(f"== {d}: {len(runs)} of {len(planned)} planned runs")
    for name, rc in stages:
        if rc == "124":
            out.append(f"   stage {name}: TIMEOUT (stage cut off)")
            bad.append(f"{os.path.basename(d)}: stage TIMEOUT")
        elif rc != "0":
            out.append(f"   stage {name}: rc={rc}")
            bad.append(f"{os.path.basename(d)}: stage rc={rc}")
    done = {(k, f) for k, f, _, _ in runs}
    for k, f in planned:
        if (k, f) not in done:
            out.append(f"   {k:<10} {f}  NOT RUN (planned, no result: the stage ended first)")
            bad.append(f"{k} {f}: NOT RUN")
    if planned and not stages:
        bad.append(f"{os.path.basename(d)}: no stage exit line")
    if none:
        runs.append(("none", "-", "NONE", "0"))
    if not runs:
        out.append("   NOT RUN (no status line: stage killed or never started)")
        bad.append(f"{os.path.basename(d)}: NOT RUN")
    for kind, f, rc, secs in runs:
        flags = []
        if rc == "NONE":
            out.append(f"   {f}: no test files for this stage in the overlay (UNCOVERED)")
            bad.append(f"{os.path.basename(d)}: UNCOVERED (no test files)")
            continue
        if rc == "124":
            flags.append("TIMEOUT")
        line = f"   {kind:<10} {f} rc={rc} {secs}s"
        if kind == "pytest":
            xml = os.path.join(d, os.path.basename(f)[:-3] + ".xml")
            if rc == "5":
                flags.append("NO-TESTS")
            if os.path.isfile(xml):
                try:
                    counts, failing, skips = junit(xml)
                except ET.ParseError as e:
                    counts, failing, skips = collections.Counter(), [], collections.Counter()
                    flags.append(f"BAD-JUNIT({e})")
                total = sum(counts.values())
                line += (f": {counts['passed']} passed, {counts['failed']} failed, {counts['errors']} errors, "
                         f"{counts['skipped']} skipped")
                if counts["failed"] or counts["errors"] or rc not in ("0", "5", "124"):
                    flags.append("FAIL")
                if total and counts["skipped"] == total:
                    flags.append("ALL-SKIPPED")
                for t in failing[:20]:
                    line += f"\n      failing {t}"
                for reason, n in skips.most_common(8):
                    line += f"\n      skipped x{n}: {reason}"
            else:
                flags.append("NO-JUNIT")
                if rc != "0":
                    flags.append("FAIL")
        elif rc != "0":
            flags.append("FAIL")
        out.append(line + ("  " + " ".join(flags) if flags else ""))
        if flags:
            bad.append(f"{kind} {f}: {' '.join(flags)}")
    return out, bad


def main():
    bad = []
    for d in sys.argv[1:]:
        lines, b = stage(d)
        print("\n".join(lines))
        bad += b
    print(f"GPUTEST {'FLAGGED: ' + '; '.join(bad) if bad else 'OK (every run passed; see skips above)'}")


if __name__ == "__main__":
    main()
