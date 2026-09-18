# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Anthropic's performance-engineering take-home. The single deliverable is `KernelBuilder.build_kernel` in `perf_takehome.py`: emit instructions for a simulated VLIW/SIMD machine that compute the same result as `reference_kernel2`, in as few simulated cycles as possible. Starting point is 147734 cycles; thresholds to beat are listed in `tests/submission_tests.py` (lowest is 1363).

## Hard rules

- **Never modify anything under `tests/`.** `tests/frozen_problem.py` is a frozen copy of `problem.py` and `tests/submission_tests.py` is the grader. `git diff origin/main tests/` must stay empty. The README explicitly calls out LLM agents "fixing" the tests as invalid submissions.
- **Do not change `N_CORES`** (it is 1 on purpose) or otherwise edit `problem.py` to make the problem easier. Only `perf_takehome.py` should change for a submission; edits to `problem.py` won't affect grading anyway since the grader imports `frozen_problem`.
- The cycle count that matters is the one printed by `python3 tests/submission_tests.py`, not by `perf_takehome.py`.

## Commands

`python` is not on PATH on this machine; use `python3`. No dependencies beyond the stdlib.

```bash
python3 tests/submission_tests.py            # authoritative correctness + cycle count (~2s)
python3 perf_takehome.py                     # dev tests (uses problem.py, honors debug/pause slots)
python3 perf_takehome.py Tests.test_kernel_cycles
python3 perf_takehome.py Tests.test_kernel_trace   # also writes trace.json
python3 watch_trace.py                       # serves trace.json at localhost:8000 with a Perfetto viewer (Chrome only)
```

Verification before submitting: `git diff origin/main tests/` (must be empty) and `python3 tests/submission_tests.py`.

## Architecture

`problem.py` is the simulator, ISA, reference implementation, and memory layout in one file; the top of `Machine` has the ISA overview and the individual `alu`/`valu`/`load`/`store`/`flow` methods are the full ISA spec.

**Machine model**
- One instruction bundle per cycle. A bundle is `dict[engine, list[slot]]`; `SLOT_LIMITS` caps slots per engine per cycle (`alu` 12, `valu` 6, `load` 2, `store` 2, `flow` 1, `debug` 64). All slots in a bundle read state at the start of the cycle and writes land at the end, so slots within a bundle are independent of each other.
- Bundles containing only `debug` slots don't cost a cycle. `debug` and `pause` are ignored entirely by the grader (`enable_debug = enable_pause = False`).
- Scratch (`SCRATCH_SIZE = 1536` words) is the register file / constant pool / cache. Almost every number in a slot is a scratch address; exceptions are `const`, `add_imm`'s immediate, and jump targets.
- Vector ops are `VLEN = 8` wide over contiguous scratch. `vload`/`vstore` are contiguous-only; gathers must be done with scalar `load`s (`load_offset` exists to write into an element of a vector destination).

**Problem** (`reference_kernel`): for each of `rounds` rounds, each of `batch_size` lanes reads a tree node value at its current index, hashes `val ^ node_val` through the six `HASH_STAGES`, and steps to child `2*idx + (1 or 2)` depending on parity, wrapping to root when past `n_nodes`. Grading config is `forest_height=10, rounds=16, batch_size=256`; only final `inp_values` in memory is checked (indices need not be written back). Memory layout is in `build_mem_image` — header at `mem[0..7]` holds sizes and the three region pointers; `mem[7]` points at spare scratch memory after the inputs.

**Kernel builder** (`perf_takehome.py`): `KernelBuilder.build_kernel` emits an SSA IR (`_VReg`/`_Op`, each op an engine + slot template) and `_Sched` list-schedules it into bundles, allocating physical scratch when an op is scheduled and freeing it at a vreg's last use (reads happen at cycle start, writes at end, so a reader and a new writer can share a cycle). Memory ordering is not tracked — pass an unrendered token vreg as an extra source where stores must precede loads (see `tok`). Key algorithmic choices: the hash is fused to 11 valu ops/round (`multiply_add` for the `a + C + (a<<k)` stages, stages 3+4 share an input, `^C6` folded into node constants); rounds visit tree depths `[0..10, 0..4]` because every lane starts at the root, so shallow depths select node values from broadcast constants (Möbius-transform coefficients, Horner `multiply_add` / flow `vselect` trees) and deep depths gather with `load_offset`. Tuning knobs are class attributes (`GATHER_*`, `VSEL*`, `ALU_FRAC`, `PRIO_*`, `PREXOR_LEVELS`); `kb.stats` reports per-engine utilization after a build. Current result ≈1012 cycles with every engine 84–97% busy, so further gains need fewer total ops, not better packing.

**Two test harnesses**
- `perf_takehome.do_kernel_test` steps the machine in lockstep with the `reference_kernel2` generator: each `pause` slot in the program must match a `yield` in the reference, and `("debug", ("compare", addr, key))` / `vcompare` slots assert scratch values against `value_trace` at that point. Useful while developing; if you change the kernel's structure, drop or adjust the compare/pause slots accordingly or this harness will fail even when the submission harness passes.
- `tests/submission_tests.py` runs the whole program unpaused on unseeded random inputs (8 correctness runs + one timed run) and only checks final memory.
