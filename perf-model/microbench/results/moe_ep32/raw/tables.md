
### bench (capacity = T, knob cache), nvfp4: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP28 | EP32 | EP32/EP16 |
|---|---|---|---|---|
| 1 | 51.4 [51.2-51.7] (51.4) | 50.5 [50.4-50.7] (50.5) | 50.4 [50.1-50.6] (50.3/50.4) | 0.979 |
| 4 | 76.1 [75.9-76.2] (76.1) | - | 59.3 [59.1-64.9] (59.2/59.3) | 0.779 |
| 16 | 85.4 [85.3-85.6] (85.4) | 62.5 [62.5-62.8] (62.5) | 59.4 [59.2-59.7] (59.5/59.3) | 0.696 |
| 32 | 85.8 [85.6-86.0] (85.8) | - | 60.2 [60.1-60.5] (60.3/60.1) | 0.702 |
| 64 | 85.8 [85.7-85.9] (85.8) | - | 62.8 [62.7-63.0] (62.8/62.8) | 0.732 |
| 128 | 88.1 [88.0-88.3] (88.1) | 71.2 [71.1-71.3] (71.2) | 69.1 [68.9-69.6] (68.9/69.3) | 0.784 |
| 256 | 99.6 [99.5-99.8] (99.6) | - | 94.5 [94.3-95.1] (94.6/94.5) | 0.948 |
| 512 | 133.4 [133.3-133.6] (133.4) | 140.8 [140.6-142.5] (140.8) | 143.3 [143.2-143.5] (143.3/143.3) | 1.074 |

### bench (capacity = T, knob cache), mxfp4_mxfp8: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP28 | EP32 | EP32/EP16 |
|---|---|---|---|---|
| 1 | 54.0 [53.7-54.3] (54.0) | 52.8 [52.7-53.0] (52.8) | 53.4 [53.3-53.6] (53.4/53.5) | 0.989 |
| 4 | 77.9 [77.7-78.0] (77.9) | - | 62.9 [62.8-63.2] (63.0/62.9) | 0.808 |
| 16 | 90.3 [90.1-90.4] (90.3) | 67.4 [67.3-67.5] (67.4) | 65.2 [65.1-65.4] (65.2/65.2) | 0.722 |
| 32 | 91.4 [91.2-91.6] (91.4) | - | 66.6 [66.5-66.9] (66.6/66.7) | 0.729 |
| 64 | 93.8 [93.7-94.0] (93.8) | - | 70.3 [70.2-70.7] (70.3/70.3) | 0.749 |
| 128 | 96.8 [96.6-96.9] (96.8) | 79.2 [79.0-79.3] (79.2) | 77.5 [77.1-78.1] (77.1/77.9) | 0.801 |
| 256 | 105.5 [105.3-105.6] (105.5) | - | 101.1 [100.9-101.2] (101.1/101.1) | 0.958 |
| 512 | 154.2 [146.6-156.8] (154.2) | 151.2 [150.3-153.1] (151.2) | 156.8 [156.5-157.2] (156.9/156.7) | 1.017 |

### vLLM (capacity 8192, knobs=None, prepare([T]), hint T), nvfp4: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP32 | EP32/EP16 |
|---|---|---|---|
| 1 | 51.5 [51.2-51.7] (51.5) | 50.4 [50.2-50.5] (50.4/50.3) | 0.978 |
| 4 | 76.0 [75.9-76.2] (76.0) | 59.1 [59.0-59.3] (59.1/59.1) | 0.778 |
| 16 | 85.1 [84.9-85.3] (85.1) | 59.2 [59.1-65.8] (59.2/59.2) | 0.696 |
| 32 | 84.8 [84.7-84.9] (84.8) | 61.1 [61.0-61.3] (61.1/61.1) | 0.720 |
| 64 | 87.1 [87.0-87.3] (87.1) | 61.8 [61.7-62.0] (61.8/61.8) | 0.710 |
| 128 | 89.9 [89.8-90.0] (89.9) | 71.5 [71.4-71.6] (71.5/71.5) | 0.795 |
| 256 | 101.2 [101.1-101.3] (101.2) | 98.2 [98.1-98.9] (98.2/98.2) | 0.970 |
| 512 | 134.3 [134.2-134.5] (134.3) | 143.7 [143.5-144.2] (143.7/143.6) | 1.070 |

### vLLM (capacity 8192, knobs=None, prepare([T]), hint T), mxfp4_mxfp8: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP32 | EP32/EP16 |
|---|---|---|---|
| 1 | 55.2 [54.8-55.4] (55.2) | 53.0 [52.8-53.2] (53.0/53.0) | 0.961 |
| 4 | 77.2 [77.0-77.3] (77.2) | 62.7 [62.6-62.9] (62.7/62.7) | 0.812 |
| 16 | 88.3 [88.3-88.4] (88.3) | 64.4 [64.3-64.6] (64.4/64.4) | 0.728 |
| 32 | 89.7 [89.6-89.8] (89.7) | 66.9 [66.8-67.2] (67.0/66.9) | 0.747 |
| 64 | 92.5 [92.4-92.7] (92.5) | 68.7 [68.6-68.9] (68.7/68.7) | 0.742 |
| 128 | 94.3 [94.2-94.4] (94.3) | 79.7 [79.6-80.0] (79.7/79.7) | 0.845 |
| 256 | 107.4 [107.3-112.9] (107.4) | 105.0 [104.9-105.2] (105.0/105.1) | 0.978 |
| 512 | 155.2 [148.0-156.8] (155.2) | 157.6 [157.4-158.2] (157.7/157.6) | 1.016 |

### prefill bench (capacity = T), nvfp4: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP28 | EP32 | EP32/EP16 |
|---|---|---|---|---|
| 4096 | 651.9 [651.4-652.6] (651.9) | - | 682.1 [681.4-692.0] (682.1/682.1) | 1.046 |
| 8192 | 1255.0 [1253.2-1257.6] (1255.0) | 1423.9 [1422.8-1425.1] (1423.9) | 1373.1 [1371.6-1374.6] (1373.1/1373.1) | 1.094 |
| 16384 | 2660.5 [2657.6-2662.9] (2660.5) | - | 3464.1 [3457.0-3472.0] (3464.1) | 1.302 |

### prefill bench (capacity = T), mxfp4_mxfp8: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP28 | EP32 | EP32/EP16 |
|---|---|---|---|---|
| 4096 | 860.3 [837.0-874.4] (860.3) | - | 859.3 [839.1-867.6] (857.6/861.0) | 0.999 |
| 8192 | 1717.8 [1681.8-1735.3] (1717.8) | 1790.3 [1783.2-1795.6] (1790.3) | 1775.6 [1757.9-1783.5] (1774.5/1776.6) | 1.034 |
| 16384 | 3450.8 [3436.4-3459.9] (3450.8) | - | 4185.0 [4170.8-4197.8] (4185.0) | 1.213 |

### other ep32-* rows (C regression, X cross-EP)

- ep32-C.final.nvfp4.ep16 T1 nvfp4 ep16 status=ok med=51.440160274505615 bitwise=None maxabs=None check=pass rel_l2=0.1692078560590744
- ep32-C.final.nvfp4.ep16 T64 nvfp4 ep16 status=ok med=85.87887763977051 bitwise=None maxabs=None check=pass rel_l2=0.16648419201374054
- ep32-C.final.nvfp4.ep16 T512 nvfp4 ep16 status=ok med=133.4990406036377 bitwise=None maxabs=None check=pass rel_l2=0.16547362506389618
- ep32-C.final.mxfp4_mxfp8.ep16 T1 mxfp4_mxfp8 ep16 status=ok med=54.07248020172119 bitwise=None maxabs=None check=pass rel_l2=0.04824802652001381
- ep32-C.final.mxfp4_mxfp8.ep16 T64 mxfp4_mxfp8 ep16 status=ok med=93.85936260223389 bitwise=None maxabs=None check=pass rel_l2=0.047248683869838715
- ep32-C.final.mxfp4_mxfp8.ep16 T512 mxfp4_mxfp8 ep16 status=ok med=154.25343990325928 bitwise=None maxabs=None check=pass rel_l2=0.04729421064257622
- ep32-X.r3ep.nvfp4.ep32 T1 nvfp4 ep32 status=ok med=50.1859188079834 bitwise=None maxabs=None check=pass rel_l2=0.16932713985443115
- ep32-X.r3ep.nvfp4.ep32 T4 nvfp4 ep32 status=ok med=59.81344223022461 bitwise=None maxabs=None check=pass rel_l2=0.16726695001125336
- ep32-X.r3ep.nvfp4.ep32 T64 nvfp4 ep32 status=ok med=62.99744129180908 bitwise=None maxabs=None check=pass rel_l2=0.16604390740394592
- ep32-X.r3ep.nvfp4.ep32 T256 nvfp4 ep32 status=ok med=94.73823547363281 bitwise=None maxabs=None check=pass rel_l2=0.1659911423921585
- ep32-X.r3ep.nvfp4.ep32 T1024 nvfp4 ep32 status=ok med=219.8044776916504 bitwise=None maxabs=None check=pass rel_l2=0.16639378666877747
- ep32-X.r3ep.mxfp4_mxfp8.ep32 T1 mxfp4_mxfp8 ep32 status=ok med=53.90848159790039 bitwise=None maxabs=None check=pass rel_l2=0.04884786158800125
- ep32-X.r3ep.mxfp4_mxfp8.ep32 T4 mxfp4_mxfp8 ep32 status=ok med=63.74688148498535 bitwise=None maxabs=None check=pass rel_l2=0.047624312341213226
- ep32-X.r3ep.mxfp4_mxfp8.ep32 T64 mxfp4_mxfp8 ep32 status=ok med=70.44640064239502 bitwise=None maxabs=None check=pass rel_l2=0.04749194532632828
- ep32-X.r3ep.mxfp4_mxfp8.ep32 T256 mxfp4_mxfp8 ep32 status=ok med=100.97984313964844 bitwise=None maxabs=None check=pass rel_l2=0.04757525399327278
- ep32-X.r3ep.mxfp4_mxfp8.ep32 T1024 mxfp4_mxfp8 ep32 status=ok med=250.89727401733398 bitwise=None maxabs=None check=pass rel_l2=0.047375380992889404
- ep32-C.r3ep.nvfp4.ep16 T1 nvfp4 ep16 status=ok med=51.59567832946777 bitwise=True maxabs=0.0 check=pass rel_l2=0.1692078560590744
- ep32-C.r3ep.nvfp4.ep16 T64 nvfp4 ep16 status=ok med=85.80656051635742 bitwise=True maxabs=0.0 check=pass rel_l2=0.16648419201374054
- ep32-C.r3ep.nvfp4.ep16 T512 nvfp4 ep16 status=ok med=133.4284782409668 bitwise=True maxabs=0.0 check=pass rel_l2=0.16547362506389618
- ep32-C.r3ep.mxfp4_mxfp8.ep16 T1 mxfp4_mxfp8 ep16 status=ok med=54.06415939331055 bitwise=True maxabs=0.0 check=pass rel_l2=0.04824802652001381
- ep32-C.r3ep.mxfp4_mxfp8.ep16 T64 mxfp4_mxfp8 ep16 status=ok med=93.84063720703125 bitwise=True maxabs=0.0 check=pass rel_l2=0.047248683869838715
- ep32-C.r3ep.mxfp4_mxfp8.ep16 T512 mxfp4_mxfp8 ep16 status=ok med=154.1820764541626 bitwise=True maxabs=0.0 check=pass rel_l2=0.04729421064257622
- ep32-X.r3ep.mxfp4_mxfp8.ep16 T2 mxfp4_mxfp8 ep16 status=ok med=63.69855880737305 bitwise=None maxabs=None check=pass rel_l2=0.04740007966756821
- ep32-X.r3ep.mxfp4_mxfp8.ep16 T8 mxfp4_mxfp8 ep16 status=ok med=87.06912040710449 bitwise=None maxabs=None check=pass rel_l2=0.04754530265927315
- ep32-X.r3ep.mxfp4_mxfp8.ep16 T128 mxfp4_mxfp8 ep16 status=ok med=96.72767639160156 bitwise=None maxabs=None check=pass rel_l2=0.04749194532632828
- ep32-X.r3ep.mxfp4_mxfp8.ep16 T512 mxfp4_mxfp8 ep16 status=ok med=144.85055923461914 bitwise=None maxabs=None check=pass rel_l2=0.04757525399327278
- ep32-X.r3ep.mxfp4_mxfp8.ep16 T2048 mxfp4_mxfp8 ep16 status=ok med=455.53760528564453 bitwise=None maxabs=None check=pass rel_l2=0.047375380992889404
- ep32-X.r3ep.mxfp4_mxfp8.ep16 T7 mxfp4_mxfp8 ep16 status=ok med=86.52768135070801 bitwise=None maxabs=None check=pass rel_l2=0.04746146872639656
- ep32-X.r3ep.mxfp4_mxfp8.ep16 T112 mxfp4_mxfp8 ep16 status=ok med=95.37535667419434 bitwise=None maxabs=None check=pass rel_l2=0.047097623348236084
- ep32-X.r3ep.mxfp4_mxfp8.ep16 T448 mxfp4_mxfp8 ep16 status=ok med=135.81279754638672 bitwise=None maxabs=None check=pass rel_l2=0.047515593469142914
- ep32-X.r3ep.nvfp4.ep16 T2 nvfp4 ep16 status=ok med=60.916481018066406 bitwise=None maxabs=None check=pass rel_l2=0.1667049080133438
- ep32-X.r3ep.nvfp4.ep16 T8 nvfp4 ep16 status=ok med=82.67359733581543 bitwise=None maxabs=None check=pass rel_l2=0.16590414941310883
- ep32-X.r3ep.nvfp4.ep16 T128 nvfp4 ep16 status=ok med=88.41888427734375 bitwise=None maxabs=None check=pass rel_l2=0.16604390740394592
- ep32-X.r3ep.nvfp4.ep16 T512 nvfp4 ep16 status=ok med=132.8112030029297 bitwise=None maxabs=None check=pass rel_l2=0.16563504934310913
- ep32-X.r3ep.nvfp4.ep16 T2048 nvfp4 ep16 status=ok med=355.02464294433594 bitwise=None maxabs=None check=pass rel_l2=0.16639378666877747
- ep32-X.r3ep.nvfp4.ep16 T7 nvfp4 ep16 status=ok med=82.4953556060791 bitwise=None maxabs=None check=pass rel_l2=0.1658560335636139
- ep32-X.r3ep.nvfp4.ep16 T112 nvfp4 ep16 status=ok med=87.90143966674805 bitwise=None maxabs=None check=pass rel_l2=0.1651887446641922
- ep32-X.r3ep.nvfp4.ep16 T448 nvfp4 ep16 status=ok med=125.27168273925781 bitwise=None maxabs=None check=pass rel_l2=0.16646021604537964
- ep32-X.r3ep.nvfp4.ep28 T4 nvfp4 ep28 status=ok med=63.24480056762695 bitwise=None maxabs=None check=pass rel_l2=0.16745515167713165
- ep32-X.r3ep.nvfp4.ep28 T64 nvfp4 ep28 status=ok med=65.63648223876953 bitwise=None maxabs=None check=pass rel_l2=0.16643333435058594
- ep32-X.r3ep.nvfp4.ep28 T256 nvfp4 ep28 status=ok med=93.80288124084473 bitwise=None maxabs=None check=pass rel_l2=0.16646023094654083
- ep32-X.r3ep.mxfp4_mxfp8.ep28 T4 mxfp4_mxfp8 ep28 status=ok med=66.31455898284912 bitwise=None maxabs=None check=pass rel_l2=0.04772206395864487
- ep32-X.r3ep.mxfp4_mxfp8.ep28 T64 mxfp4_mxfp8 ep28 status=ok med=72.95231819152832 bitwise=None maxabs=None check=pass rel_l2=0.047622546553611755
- ep32-X.r3ep.mxfp4_mxfp8.ep28 T256 mxfp4_mxfp8 ep28 status=ok med=101.86528205871582 bitwise=None maxabs=None check=pass rel_l2=0.047308582812547684
- ep32-pf16g.r3ep.nvfp4.ep32.r1 T8192 nvfp4 ep32 status=ok med=1373.1766510009766 bitwise=None maxabs=None check=pass rel_l2=0.16528087854385376
- ep32-pf16g.r3ep.mxfp4_mxfp8.ep32.r1 T8192 mxfp4_mxfp8 ep32 status=ok med=1775.169448852539 bitwise=None maxabs=None check=pass rel_l2=0.047195613384246826
- ep32-pf16g.r3ep.mxfp4_mxfp8.ep32.t16k T16384 mxfp4_mxfp8 ep32 status=ok med=4183.831329345703 bitwise=None maxabs=None check=pass rel_l2=0.04746130853891373
