#!/bin/bash
# D-KDA round 7 (c): is the producer the bottleneck of the tensor-memory arms? New producer
# (claims issued one item ahead) + a separate copier warp (tcgen05.cp + commits), for the
# math arms ('j', MATH 11) and a no-math TMEM-handshake arm ('y', MATH 12); the old stream
# (old producer) as reference. 8 KiB operand block on the slot, die-local.
B=benchmarks/kernels/k31_vr/kda_skel.py
timeout 900 python3 $B --n 32 64 96 128 --place local --sin 8192 --arms rmwt512o1y rmwt256o1y rmwt512o1j rmwt256o1j rmwt512o1 2>&1 | grep -E "us/layer|regs|rror|check" | sed "s/ (R .*ok=/ ok=/" | cut -c1-150
