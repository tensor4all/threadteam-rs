# Confirmation of the dispatch-cost gates (rough, one M5 Max)

Probe: `src/confirm.rs` (built against the dispatch-cost branches), 3 processes x 2 rounds, medians, 6 threads, QoS UserInitiated, 300 ms settle per cell, `src/runconfirm.sh` under `hostrun.py` (log: `raw/processes.log`, tags `confirm-p1..3`). Idle before each process 86.2, 85.2, 87.9 percent, after 86.9, 88.9, 85.6; load average (1 min) 9.2, 3.9, 4.2; top 3 consumers were WindowServer, Claude Helper and TextEdit / azooKeyMac / Codex renderer (no benchmark or cargo). All three are marked busy: the idle sample at about 150 s was 72.4, 74.7 and 68.4 percent. Ratios are mode / serial; below 1 is a win.

| family | size | serial | plain_out/seq | plain_in/seq | team/seq |
| --- | --- | --- | --- | --- | --- |
| copy | 1024 | 0.1 us | 1.06 | 1.07 | 0.88 |
| copy | 2048 | 0.2 us | 1.09 | 1.10 | 1.08 |
| copy | 4096 | 0.4 us | 1.09 | 1.07 | 1.13 |
| copy | 8192 | 0.8 us | 0.81 | 1.00 | 0.95 |
| copy | 16384 | 1.4 us | 1.11 | 1.13 | 1.01 |
| copy | 32768 | 3.9 us | 1.01 | 0.98 | 0.95 |
| copy | 65536 | 8.0 us | 1.01 | 0.97 | 0.97 |
| copy | 131072 | 15.9 us | 0.98 | 0.98 | 0.84 |
| copy | 262144 | 31.7 us | 0.90 | 0.85 | 0.37 |
| copy | 524288 | 70.1 us | 0.61 | 0.58 | 0.30 |
| copy | 1048576 | 145.2 us | 0.50 | 0.52 | 0.31 |
| copy | 2097152 | 292.6 us | 0.50 | 0.49 | 0.32 |
| einsum_chain | 32 | 3.7 us | 1.03 | 0.92 | 0.81 |
| einsum_chain | 48 | 11.4 us | 1.09 | 0.92 | 1.02 |
| einsum_chain | 64 | 28.4 us | 1.01 | 1.00 | 0.29 |
| einsum_chain | 96 | 88.4 us | 1.08 | 0.93 | 0.24 |
| einsum_chain | 128 | 204.7 us | 0.57 | 0.33 | 0.22 |
| einsum_chain | 192 | 743.0 us | 0.28 | 0.22 | 0.17 |
| exp | 1024 | 1.9 us | 0.96 | 0.74 | 0.77 |
| exp | 2048 | 3.8 us | 1.02 | 0.94 | 0.98 |
| exp | 4096 | 7.6 us | 1.02 | 1.03 | 1.03 |
| exp | 8192 | 14.9 us | 1.03 | 0.94 | 0.99 |
| exp | 16384 | 30.5 us | 0.99 | 0.97 | 1.01 |
| exp | 32768 | 61.1 us | 0.97 | 0.99 | 0.97 |
| exp | 65536 | 120.6 us | 1.02 | 0.97 | 1.04 |
| exp | 131072 | 243.9 us | 0.97 | 0.97 | 0.50 |
| exp | 262144 | 490.4 us | 1.00 | 0.50 | 0.24 |
| exp | 524288 | 962.3 us | 0.30 | 0.26 | 0.25 |
| exp | 1048576 | 2329.9 us | 0.23 | 0.22 | 0.19 |
| exp | 2097152 | 4058.3 us | 0.28 | 0.27 | 0.20 |
| map | 1024 | 0.1 us | 1.03 | 1.03 | 0.90 |
| map | 2048 | 0.2 us | 1.11 | 1.11 | 1.03 |
| map | 4096 | 0.4 us | 1.32 | 1.34 | 1.12 |
| map | 8192 | 1.1 us | 0.97 | 1.04 | 0.94 |
| map | 16384 | 1.8 us | 1.01 | 1.03 | 1.00 |
| map | 32768 | 3.9 us | 1.01 | 1.00 | 0.90 |
| map | 65536 | 7.9 us | 0.99 | 1.00 | 0.92 |
| map | 131072 | 16.0 us | 1.00 | 0.99 | 0.73 |
| map | 262144 | 33.7 us | 1.01 | 0.75 | 0.35 |
| map | 524288 | 71.1 us | 0.68 | 0.54 | 0.31 |
| map | 1048576 | 159.3 us | 0.45 | 0.37 | 0.27 |
| map | 2097152 | 393.1 us | 0.40 | 0.29 | 0.22 |
| sum | 1024 | 0.1 us | 1.03 | 0.97 | 0.78 |
| sum | 2048 | 0.2 us | 1.03 | 0.94 | 0.75 |
| sum | 4096 | 0.4 us | 1.03 | 0.97 | 0.95 |
| sum | 8192 | 0.9 us | 1.01 | 0.99 | 0.89 |
| sum | 16384 | 1.9 us | 0.93 | 0.91 | 0.99 |
| sum | 32768 | 3.6 us | 0.97 | 1.02 | 1.02 |
| sum | 65536 | 5.8 us | 1.26 | 1.12 | 1.27 |
| sum | 131072 | 13.0 us | 1.16 | 1.05 | 0.80 |
| sum | 262144 | 28.3 us | 1.06 | 0.74 | 0.20 |
| sum | 524288 | 60.0 us | 0.88 | 0.59 | 0.17 |
| sum | 1048576 | 115.7 us | 0.66 | 0.24 | 0.17 |
| sum | 2097152 | 234.1 us | 0.38 | 0.26 | 0.18 |
| tc_matmul | 32 | 1.9 us | 1.02 | 0.89 | 0.78 |
| tc_matmul | 48 | 5.5 us | 1.08 | 0.94 | 1.16 |
| tc_matmul | 64 | 11.7 us | 1.24 | 1.03 | 0.35 |
| tc_matmul | 96 | 36.5 us | 1.31 | 1.12 | 0.22 |
| tc_matmul | 128 | 92.4 us | 0.61 | 0.37 | 0.24 |
| tc_matmul | 192 | 310.6 us | 0.33 | 0.29 | 0.20 |
| tc_matmul | 256 | 666.3 us | 0.31 | 0.24 | 0.20 |
