"""Variant generator: run the fixed-route chain (GATE2 fused gate, wide V
residual, FIN-gated epilogue image, per-element partial tail) on every shape.

usage: mkg2.py IN OUT [ROWPAIR]
Forces GATE2_=1, QK_ROWPAIR_=ROWPAIR (default 4), GATE_ROLL_=0,
PARTIAL_TMA_=0 inside _launch_pkd, whatever the host route passes.
"""
import sys

src = open(sys.argv[1]).read()
rp = int(sys.argv[3]) if len(sys.argv) > 3 else 4
i = src.index("def _launch_pkd(")
j = src.index("    CLUSTER_: cutlass.Constexpr[int] = 1,\n):\n", i)
j += len("    CLUSTER_: cutlass.Constexpr[int] = 1,\n):\n")
ov = (
    "    GATE2_ = 1\n"
    f"    QK_ROWPAIR_ = {rp}\n"
    "    GATE_ROLL_ = 0\n"
    "    PARTIAL_TMA_ = 0\n"
)
# keep a leading docstring (if any) first
body = src[j:]
if body.lstrip().startswith('"""'):
    k = body.index('"""', body.index('"""') + 3) + 3
    k = body.index("\n", k) + 1
    body = body[:k] + ov + body[k:]
else:
    body = ov + body
open(sys.argv[2], "w").write(src[:j] + body)
print("ok")
