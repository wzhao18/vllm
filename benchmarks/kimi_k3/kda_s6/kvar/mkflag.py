"""Copy kernel.py with module flags set: mkflag.py IN OUT NAME=VALUE [NAME=VALUE ...]
(e.g. DP_PREP=1 PREP_ONLY=1)."""
import re
import sys

src = open(sys.argv[1]).read()
for kv in sys.argv[3:]:
    name, val = kv.split("=")
    src, n = re.subn(rf"^{name} = \S+$", f"{name} = {val}", src, count=1, flags=re.M)
    assert n == 1, name
open(sys.argv[2], "w").write(src)
