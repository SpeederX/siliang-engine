# DS4 maintainer continuity baseline — 2026-09-02

Primary continuity cell: `llama-server`, K216/R12/P12, L2 8 GiB LRU, L1 SLFU, `admit-k-cold=on`, `demote-k-hot=off`, DeepSeek4 FRONT roll, routed prefill off, 2 CPU threads, 256 decode tokens, fresh process per replica. Current v0.1.6 managed/no-mmap backing is used.

Current 3-rep median: **1.880 tok/s** server timing; **1.873 tok/s** cumulative at token 256; **1.818 tok/s** over the 128→256 window. Median route state at 256: **25.56% K / 36.77% L2 / 36.74% cold**.

This natural-sampling cell measures maintainer workload variance; it is not the deterministic correctness gate. A separate deterministic temp=0/seed=1 gate should be used for byte/token regression.

The performance suite will grow to three fixed workload fixtures (prose, code, hybrid) while retaining this historical prompt as a continuity arm.
