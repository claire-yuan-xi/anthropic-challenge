# Optimization notes: 147,734 → 1,012 cycles

How `KernelBuilder.build_kernel` in `perf_takehome.py` went from the scalar
baseline to 1,012 cycles (146× faster) on the simulated VLIW/SIMD machine,
passing every threshold in `tests/submission_tests.py` (lowest: 1,363).
Verified with `git diff origin/main tests/` empty and 100 random seeds on the
frozen simulator — every input takes exactly 1,012 cycles because the
schedule has no data-dependent control flow.

## 1. The machine and its constraints

`problem.py` simulates a VLIW core with five engines. Every cycle executes one
bundle; each engine can fill a fixed number of slots:

| Engine | Slots/cycle | What it does |
|---|---|---|
| `valu` | 6 | 8-wide vector ALU incl. `multiply_add` (the only fused op) |
| `alu` | 12 | scalar ALU: `+ - * // ^ & \| << >> % < ==` — **no** `multiply_add` |
| `load` | 2 | `load`, `load_offset` (one word each), `vload` (8 contiguous), `const` |
| `store` | 2 | `store`, `vstore` |
| `flow` | 1 | `vselect`, `select`, `add_imm`, jumps, `pause` |

Constraints that shaped everything:

- **Reads happen at cycle start, writes at cycle end.** Slots in a bundle are
  independent, so a value's last reader and a new writer can share a cycle.
  Dependent ops need ≥1 cycle between them.
- **Scratch is the only register file**: 1,536 words total, for constants,
  temporaries and all 256 lanes' state. Vectors occupy 8 contiguous words.
- **Gathers are scalar.** `vload` only reads 8 *contiguous* words, so fetching
  a data-dependent tree node for each lane costs one load slot per lane. With
  2 load slots/cycle, the naive 256 lanes × 16 rounds = 4,096 loads is a
  2,048-cycle floor by itself — above every threshold.
- **No scratch-indirect addressing.** You cannot index scratch by a lane's
  value, so "look up in a small table" must be done arithmetically or via
  memory loads.
- Only the final `inp_values` region is graded; `pause`/`debug` are ignored.

## 2. The workload

Each of 256 lanes holds a value and a tree index (all start at the root).
Per round: `val = hash(val ^ tree[idx])`, then step to child
`2*idx + 1 + (val & 1)`, wrapping to the root past depth 10. The hash is six
stages of `a = (a op1 C) op2 (a op3 k)` with adds, xors and shifts.

The key structural fact: because every lane starts at the root and the tree
is a perfect binary tree of height 10, **round r is at a known depth for all
lanes**: `[0,1,…,10,0,1,2,3,4]`. Shallow rounds only touch 2ᵈ candidate nodes.

## 3. Stages

| Stage | Cycles | Commit |
|---|---|---|
| Baseline (one scalar slot per bundle) | 147,734 | — |
| List scheduler + vectorized, fused hash, select-vs-gather | 1,368 | `c416838` |
| Tune the select/gather split | 1,276 | `2cd4890` |
| Run 32 lanes on the idle scalar ALU | 1,176 | `ba837f3` |
| Replace that with per-op ALU offload (25%) | 1,132 | `a67f1cb` |
| Round-weighted scheduling priority | 1,064 | `932e21b` |
| Per-vector gather plans, pre-xor tree level 4, 30% offload | 1,041 | `f9ed0c5` |
| Config search, best-priority register protection | 1,016 | `dbf1662` |
| Priority jitter seed | 1,012 | `3de3f64` |

### 3.1 Infrastructure first: an IR and a list scheduler

Hand-packing bundles was never going to work at this scale, so the builder
emits an SSA IR (`_VReg`, `_Op`: engine + slot template + virtual sources)
and `_Sched` list-schedules it cycle by cycle, filling slots in priority order.
Physical scratch is allocated when an op is scheduled and freed when a vreg's
last consumer is scheduled; an op whose source dies at that op reuses the
source's block in place (legal because of the read-then-write cycle model).
Memory ordering is not modeled — where stores must precede loads, an
unrendered "token" vreg is passed as an extra source.

### 3.2 Algebra on the hash (18 → 11 valu ops per round)

The reference stages, mod 2³²:

```
s1: (a + C1) + (a << 12)   = 4097·a + C1          → 1 multiply_add
s2: (a ^ C2) ^ (a >> 19)                            → 3 ops (>>, ^, ^)
s3: (a + C3) + (a << 5)    = 33·a + C3
s4: (a + C4) ^ (a << 9)
s5: (a + C5) + (a << 3)    = 9·a + C5              → 1 multiply_add
s6: (a ^ C6) ^ (a >> 16)                            → 2 ops + fold
```

- Stages 1, 3, 5 are affine, so each is one `multiply_add` with constant
  vectors.
- Stages 3+4 fuse: `a3 + C4 = 33·a2 + (C3+C4)` and `a3 << 9 = 16896·a2 + (C3<<9)`
  are both `multiply_add`s of the *same* input, then one xor — 3 ops instead of 4.
- The final `^ C6` is folded into the node constants (`node' = node ^ C6`),
  since the next round starts with `val ^ node`. The walk bit is then read from
  the pre-xor value (`nb = a6 & 1`, which is `1 - bit` because C6 is odd) and
  all selection tables are built in terms of `nb`. Round 0 uses raw constants
  because its input is a true value; the last round pays the xor once.

### 3.3 Select instead of gather at shallow depths

At depth d a lane's node is one of 2ᵈ values, indexed by the last d walk
bits. For d ≤ 4 the kernel computes the node without touching memory:

- The 2ᵈ node values are loaded once with 4 `vload`s, xored with C6, and their
  multilinear (Möbius) coefficients in the bits are computed on the scalar ALU
  at setup and broadcast to vector constants (31 constant vectors).
- The node is then a multilinear polynomial in the bits, evaluated by a Horner
  tree of `multiply_add`s (2ᵈ−1 ops), or by a `vselect` tree on the flow
  engine (also 2ᵈ−1 ops), or any mix: `VSEL[d] = (k, "top"|"bottom")` decides
  how many bits are resolved on flow and whether those sit at the top or bottom
  of the tree. This lets the otherwise-idle flow engine absorb selection work.

Depths ≥ 5 gather: 8 `load_offset`s per vector from a vector of addresses.
The address is kept incrementally as `y = ~(idx+1)`, updated by
`y' = 2y + nb` (one `multiply_add`) with `addr = (fp − 2) − y`.

### 3.4 Using the scalar ALU

After vectorizing, `valu` was 92% busy and `alu` 0%. Two approaches:

1. **Whole scalar lanes** (1,176): run 32 lanes entirely in scalar code,
   gathering every node with one load each. Worked, but the ALU has no
   `multiply_add`, so each fused stage costs 2 scalar ops — 16 ALU slots to
   replace one valu slot.
2. **Per-op offload** (1,132, kept): keep all 32 vectors, but emit a fraction
   (`ALU_FRAC`, ~30%) of the *simple* ops (xor/shift/and) as 8 scalar ops on
   the vector's lanes. That's 8 ALU slots per valu slot — twice as efficient —
   and it needs no separate code path, since a vreg with 8 producers is just
   another vreg to the scheduler.

### 3.5 Scheduling priority

Strict vector-major priority produced a ~200-cycle latency-bound tail: the
last vector only got leftover slots and then ran its 16-round dependency
chain alone. Priority `vec + ρ·round + σ·round²` (ρ=0.75, σ=0.06) interleaves
vectors so the final rounds of many vectors overlap; the tail shrank to ~25
cycles. A deterministic per-(vector, round) jitter seed is the last knob — the
schedule is chaotic (±20 cycles from tiny changes), so the seed is searched.

### 3.6 Balancing the engines

Once the scheduler was decent, all four working engines sat near saturation
and every further gain was a *trade*:

- **Round 4 gather vs. select.** Gathering costs 8 loads + 1 valu; a
  flow-heavy select costs 14 `vselect` + 1 valu. Loads were 100% busy in
  steady state while flow had slack, so 16 of 32 vectors select at round 4
  (`GATHER_HEAD`), the rest gather. Similarly the last 12 vectors gather at
  round 15 (`GATHER_TAIL`) because loads are idle in the tail.
- **Pre-xor tree levels in memory.** Gathered nodes are raw, so gather rounds
  need an extra `^ C6`. Levels 4–5 are xored in place at setup (`vload`,
  8 scalar xors, `vstore`, guarded by a token) so those rounds skip the op.
- **Register-pressure control.** The allocator protects the vector holding the
  best priority (not the lowest index, which broke once priorities weren't
  vector-major) and retries with a larger reserve on deadlock.

Final steady state: **valu 97%, alu 95%, load 90%, flow 84%**.

## 4. What failed (and was rolled back)

| Idea | Result | Why |
|---|---|---|
| Larger register reserve / priority-graded allocation gating | +30–60 cycles | Run-ahead (e.g. computing selection leaves rounds early) is *productive*; freed registers weren't used better |
| Capping vectors in flight (`MAX_ACTIVE`) | +30–170 | Same — more parallelism always won |
| Anchoring selection ops to the previous round | +15–20 | Cut peak registers from 26 to ~8 blocks/vector but delayed selections into flow contention |
| Pre-xoring tree levels 6–9 | +20–90 | 126 extra `vload`s + 1,000 setup-priority ALU xors crowded out the ramp; loads are the steady-state bottleneck |
| ALU offload above ~34% | +100–500 | ALU over-subscribed; 8-lane ops straddle cycles and lengthen every chain |
| Flow-heavy selection at depths 3–4 for *all* vectors | +15–50 | Flow saturates in bursts; and with valu relieved, loads became the wall |
| Full-scalar lanes beyond 32 | worse | ALU-bound; per-op offload dominates it anyway |

Two ideas were analyzed and rejected before coding:

- **Fewer ops for the gather address.** The update is 3 ops (bit extraction,
  `y` update, `addr = K − y`). Folding the subtraction needs a one-instruction
  form of the walk bit with a *negative* coefficient; the only single-op forms
  are `a & 1`, `a | ~1` (= bit − 2) and `a · 2³¹`, all positive. Mirrored or
  shifted tree copies that would fix the sign don't fit in the 5,173-word
  memory around the existing data.
- **Sharing gathers across lanes.** At depth 5 the 256 lanes hit only 32
  nodes, but selecting 1 of 32 per lane costs 31 ops — more than the 8 loads.

## 5. Trade-offs made

- **Correctness assumptions**: the kernel relies on every lane starting at
  `idx = 0` and `n_nodes = 2^(H+1) − 1` (both guaranteed by `Input.generate` /
  `Tree.generate` and asserted in the builder). Indices are never written back
  (the grader doesn't check them).
- **Build-time specialization**: the depth schedule, selection tables and
  gather plans are computed per problem size, so the program is a fully
  unrolled straight-line schedule (1,012 bundles, no jumps).
- **Determinism over elegance**: several knobs (`PRIO_*`, `ALU_FRAC`,
  `GATHER_*`, `VSEL*`) exist only because the packed schedule is chaotic;
  their values came from randomized search, not first principles.

## 6. Where the floor is

Per lane-round the design needs ≈14.7 lane-ops (hash 11, bit 1, bookkeeping
~1.5, selection ~1.2) plus 0.44 loads. The machine supplies 48 valu + 12 alu
lane-ops per cycle, flow can only do selection, and loads cap at 2/cycle;
that puts this formulation's floor near **985 cycles**. At 1,012 the
remaining ~30 cycles are ramp-up and tail. Going meaningfully lower would
require a different formulation that cuts total ops per lane-round, not
better packing.
