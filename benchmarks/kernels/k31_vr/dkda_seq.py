# D-KDA helper: compressed opcode sequence of one SASS function (cuobjdump -sass), to see load/use
# distances. usage: python3 dkda_seq.py skel.sass <function substring>
import re
import sys

code = {"FFMA2": "F", "FFMA": "f", "LDS": "L", "SHFL": "S", "FADD": "a", "FSEL": "s", "MOV": "m",
        "STG": "G", "SYNCS": "B", "IMAD": "i", "LOP3": "o", "IADD3": "+", "BRA": "|", "STS": "T"}
on, seq = False, []
for line in open(sys.argv[1]):
    m = re.search(r"Function : (\S+)", line)
    if m:
        if on:
            break
        on = sys.argv[2] in m.group(1)
        continue
    if not on:
        continue
    m = re.search(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", line)
    if m:
        seq.append(code.get(m.group(3).split(".")[0], "."))
s = "".join(seq)
print(len(s))
for i in range(0, len(s), 100):
    print("%5d %s" % (i, s[i:i + 100]))
