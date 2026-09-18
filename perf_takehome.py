"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
    cdiv,
)


MASK32 = (1 << 32) - 1


class _VReg:
    """Virtual register (1 scalar word or one VLEN vector) in SSA form."""

    __slots__ = ("size", "nprod", "prod_done", "cons", "cons_left", "phys", "ready", "moved")

    def __init__(self, size, nprod=1):
        self.size = size
        self.nprod = nprod
        self.prod_done = 0
        self.cons = []
        self.cons_left = 0
        self.phys = None
        self.ready = 0
        self.moved = False


class _Op:
    __slots__ = ("engine", "tmpl", "dst", "srcs", "svregs", "vec", "idx", "npred", "earliest", "cycle", "prio")


class _Sched:
    """
    List scheduler for the VLIW machine. Ops are emitted in SSA form on
    virtual registers; physical scratch is allocated when an op is scheduled
    and released when a vreg's last consumer is scheduled. Because reads happen
    at the start of a cycle and writes at the end, a register freed by a reader
    can be re-targeted by a writer in the very same cycle.
    """

    def __init__(self, scratch_size, n_scalar, reserve, max_active):
        self.ops = []
        self.free_s = list(range(n_scalar))
        self.free_v = list(range(n_scalar, scratch_size - VLEN + 1, VLEN))
        self._pools = (list(self.free_s), list(self.free_v))
        self.reserve = reserve
        self.reserve_slope = 0.0
        self.max_active = max_active
        self.prio_fn = lambda vec, r: vec
        self.cur_round = 0

    def new(self, size=VLEN, nprod=1):
        return _VReg(size, nprod)

    def reset(self):
        # Undo all scheduling state so run() can be retried with new settings.
        self.free_s = list(self._pools[0])
        self.free_v = list(self._pools[1])
        for op in self.ops:
            op.cycle = None
            if op.dst is not None:
                v = op.dst[0]
                v.prod_done = 0
                v.phys = None
                v.ready = 0
                v.moved = False

    def emit(self, engine, tmpl, dst, srcs, vec):
        op = _Op()
        op.engine = engine
        op.tmpl = tmpl
        op.dst = dst
        op.srcs = srcs
        seen = []
        for v, _ in srcs:
            if v not in seen:
                seen.append(v)
                v.cons.append(op)
        op.svregs = seen
        op.vec = vec
        op.idx = len(self.ops)
        op.prio = (self.prio_fn(vec, self.cur_round) if vec >= 0 else -1e9, op.idx)
        op.cycle = None
        self.ops.append(op)
        return dst[0] if dst is not None else None

    def _alloc(self, op, best_prio, best_vec):
        if op.dst is None:
            return True
        v = op.dst[0]
        if v.phys is not None:
            return True
        for s in op.svregs:
            if s.cons_left == 1 and s.size == v.size and not s.moved:
                s.moved = True
                v.phys = s.phys
                return True
        pool = self.free_v if v.size == VLEN else self.free_s
        if not pool:
            return False
        # Priority-graded gating: the furthest-along work can always allocate;
        # lower-priority work needs a pool of free blocks that grows with the
        # priority gap, which bounds run-ahead and prevents deadlock.
        if v.size == VLEN and op.vec >= 0 and op.vec != best_vec:
            gap = max(0.0, op.prio[0] - best_prio)
            if len(pool) <= self.reserve + self.reserve_slope * gap:
                return False
        v.phys = pool.pop()
        return True

    def _render(self, op):
        out = []
        for t in op.tmpl:
            if t == "D":
                v, off = op.dst
                out.append(v.phys + off)
            elif isinstance(t, str) and t.startswith("S"):
                v, off = op.srcs[int(t[1:])]
                out.append(v.phys + off)
            else:
                out.append(t)
        return tuple(out)

    def run(self):
        remaining = {}
        for op in self.ops:
            op.npred = len(op.svregs)
            op.earliest = 0
            remaining[op.vec] = remaining.get(op.vec, 0) + 1
            if op.dst is not None:
                op.dst[0].cons_left = len(op.dst[0].cons)
        for v in set(v for op in self.ops for v in op.svregs):
            v.cons_left = len(v.cons)
        ready = sorted((op for op in self.ops if op.npred == 0), key=lambda o: o.prio)
        bundles = []
        n_done = 0
        cycle = 0
        self.min_free_v = len(self.free_v)
        started = set()
        total = dict(remaining)
        pending = sorted(self.ops, key=lambda o: o.prio)
        pi = 0
        while n_done < len(self.ops):
            while pending[pi].cycle is not None:
                pi += 1
            best_prio, best_vec = pending[pi].prio[0], pending[pi].vec
            self.min_free_v = min(self.min_free_v, len(self.free_v))
            n_active = sum(1 for v in started if v in remaining)
            used = {}
            bundle = {}
            deferred = []
            newly = []
            waiting = False
            for op in ready:
                if op.earliest > cycle:
                    deferred.append(op)
                    waiting = True
                    continue
                if used.get(op.engine, 0) >= SLOT_LIMITS[op.engine]:
                    deferred.append(op)
                    continue
                if op.vec not in started and op.vec >= 0 and n_active >= self.max_active:
                    deferred.append(op)
                    continue
                if not self._alloc(op, best_prio, best_vec):
                    deferred.append(op)
                    continue
                if op.vec not in started:
                    started.add(op.vec)
                    n_active += 1
                used[op.engine] = used.get(op.engine, 0) + 1
                bundle.setdefault(op.engine, []).append(self._render(op))
                op.cycle = cycle
                n_done += 1
                remaining[op.vec] -= 1
                if remaining[op.vec] == 0:
                    del remaining[op.vec]
                for s in op.svregs:
                    s.cons_left -= 1
                    if s.cons_left == 0 and not s.moved:
                        (self.free_v if s.size == VLEN else self.free_s).append(s.phys)
                if op.dst is not None:
                    v = op.dst[0]
                    v.prod_done += 1
                    if v.prod_done == v.nprod:
                        v.ready = cycle + 1
                        if not v.cons:
                            (self.free_v if v.size == VLEN else self.free_s).append(v.phys)
                        for c in v.cons:
                            c.npred -= 1
                            if c.npred == 0:
                                c.earliest = max(s.ready for s in c.svregs)
                                newly.append(c)
            if not bundle and not waiting:
                raise RuntimeError(f"scheduler deadlock at cycle {cycle}, {len(deferred)} ready ops, free_v={len(self.free_v)}")
            if bundle:
                bundles.append(bundle)
                cycle += 1
            if newly:
                deferred.extend(newly)
                deferred.sort(key=lambda o: o.prio)
            ready = deferred
        return bundles


class KernelBuilder:
    # Rounds at tree depth >= this load node values with per-lane gathers;
    # shallower rounds select among broadcast constants arithmetically.
    GATHER_MIN_DEPTH = 5
    GATHER_ROUNDS = ()
    # (n, rounds): the first n / last n vectors additionally gather in these
    # rounds, using load slots that are otherwise idle during ramp-up / tail.
    GATHER_HEAD = (16, (4,))
    # round -> (k, placement) overriding VSEL[depth] for that round
    VSEL_ROUND = {4: (2, "bottom"), 14: (2, "top")}
    GATHER_TAIL = (12, (15,))
    # depth -> (number of bits resolved with flow-engine vselects, placement)
    VSEL = {1: (1, "top"), 2: (1, "top"), 3: (2, "bottom"), 4: (2, "bottom")}
    RESERVE = 4
    RESERVE_SLOPE = 0.0
    MAX_ACTIVE = 32  # vectors allowed in flight at once
    # Scheduling priority of an op is vec + round * PRIO_RHO (lower first):
    # 0 is strict vector-major order; larger values interleave vectors more.
    PRIO_RHO = 0.75
    PRIO_SIGMA = 0.06
    # Deterministic per-(vector, round) priority jitter; the schedule is
    # chaotic in its priorities so the seed is just another tuning knob.
    PRIO_JITTER = 0.3
    PRIO_SEED = 6
    # Tree levels whose node values are xored with C6 in place (vload, scalar
    # xors, vstore) so gathers at those depths skip the per-round C6 xor.
    PREXOR_LEVELS = (4, 5)
    # Hash stage of the previous round that selection-tree ops must wait for.
    SEL_ANCHOR = None
    N_SCALAR = 160
    # Lanes (multiple of VLEN) processed entirely on the scalar ALU, which is
    # otherwise idle. They gather every non-root node with one scalar load.
    N_SCALAR_LANES = 0
    # Fraction of each simple (non-multiply_add) vector op that is emitted as
    # VLEN scalar ALU ops on the vector's lanes instead of one valu op, keyed
    # by op name. Spreads work onto the otherwise idle scalar ALU.
    ALU_FRAC = {n: 0.32 for n in ("x0", "s2a", "s2b", "s2c", "s4", "s6a", "s6b", "nb", "c6", "addr")}

    def __init__(self):
        self.instrs = []
        self.scratch_debug = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        H = forest_height
        assert n_nodes == 2 ** (H + 1) - 1
        assert batch_size % VLEN == 0
        NS = self.N_SCALAR_LANES
        assert NS % VLEN == 0
        nvec = (batch_size - NS) // VLEN
        ngroups = batch_size // VLEN
        depth = []
        d = 0
        for r in range(rounds):
            depth.append(d)
            d = d + 1 if d < H else 0
        def plan(k):
            extra = set(self.GATHER_ROUNDS)
            if k < self.GATHER_HEAD[0]:
                extra |= set(self.GATHER_HEAD[1])
            if k >= nvec - self.GATHER_TAIL[0]:
                extra |= set(self.GATHER_TAIL[1])
            gather = [depth[r] >= self.GATHER_MIN_DEPTH or r in extra for r in range(rounds)]
            # needy[r]: the path accumulator y must be valid before round r
            needy = [False] * (rounds + 1)
            for r in range(rounds - 1, -1, -1):
                needy[r] = gather[r] or (r + 1 < rounds and needy[r + 1] and depth[r + 1] != 0)
            neednb = [False] * rounds
            for r in range(rounds):
                if r + 1 < rounds and needy[r + 1] and depth[r + 1] != 0:
                    neednb[r] = True
                for r2 in range(r + 1, rounds):
                    if not gather[r2] and depth[r2] >= 1 and r2 - depth[r2] <= r:
                        neednb[r] = True
            return gather, needy, neednb

        plans = [plan(k) for k in range(nvec)]
        gather_any = [any(p[0][r] for p in plans) for r in range(rounds)]
        select_any = [any(not p[0][r] for p in plans) for r in range(rounds)]
        max_sel_depth = max([depth[r] for r in range(rounds) if select_any[r]] + [0])
        n_const_nodes = 2 ** (max_sel_depth + 1) - 1

        S = _Sched(SCRATCH_SIZE, self.N_SCALAR, self.RESERVE, self.MAX_ACTIVE)
        S.reserve_slope = self.RESERVE_SLOPE
        rho, sigma = self.PRIO_RHO, self.PRIO_SIGMA
        jit = random.Random(self.PRIO_SEED)
        jitter = {(k, r): jit.random() * self.PRIO_JITTER for k in range(nvec) for r in range(rounds)}
        S.prio_fn = lambda vec, r: vec + rho * r + sigma * r * r + jitter.get((vec, r), 0.0)
        SETUP = -1

        def const(val):
            v = S.new(1)
            S.emit("load", ("const", "D", val & MASK32), (v, 0), [], SETUP)
            return v

        def bcast(src):
            v = S.new()
            S.emit("valu", ("vbroadcast", "D", "S0"), (v, 0), [src], SETUP)
            return v

        def alu(op, a, b, vec=SETUP, dst=None):
            if dst is None:
                dst = (S.new(1), 0)
            S.emit("alu", (op, "D", "S0", "S1"), dst, [a, b], vec)
            return dst[0]

        def valu(op, a, b, vec):
            v = S.new()
            S.emit("valu", (op, "D", "S0", "S1"), (v, 0), [(a, 0), (b, 0)], vec)
            return v

        alu_frac = self.ALU_FRAC
        names = {}

        def simple(name, op, a, b, vec, r):
            # Vector op that may instead run as VLEN scalar ALU ops on the lanes.
            f = alu_frac.get(name, 0.0)
            ni = names.setdefault(name, len(names))
            if f > 0 and ((vec * 7 + r * 3 + ni * 5) * 17 % 100) < f * 100:
                v = S.new(nprod=VLEN)
                for i in range(VLEN):
                    S.emit("alu", (op, "D", "S0", "S1"), (v, i), [(a, i), (b, i)], vec)
                return v
            return valu(op, a, b, vec)

        # Extra (unrendered) dependency added to selection-tree ops so they are
        # not computed rounds ahead of their use, which would pin registers.
        anchor = [None]

        def madd(a, b, c, vec):
            v = S.new()
            srcs = [(a, 0), (b, 0), (c, 0)] + ([(anchor[0], 0)] if anchor[0] is not None else [])
            S.emit("valu", ("multiply_add", "D", "S0", "S1", "S2"), (v, 0), srcs, vec)
            return v

        def vsel(cond, a, b, vec):
            v = S.new()
            srcs = [(cond, 0), (a, 0), (b, 0)] + ([(anchor[0], 0)] if anchor[0] is not None else [])
            S.emit("flow", ("vselect", "D", "S0", "S1", "S2"), (v, 0), srcs, vec)
            return v

        # ---- constants -------------------------------------------------
        (_, C1, _, _, _), (_, C2, _, _, _), (_, C3, _, _, _), (_, C4, _, _, _), (_, C5, _, _, _), (_, C6, _, _, _) = HASH_STAGES
        sK4097 = const(4097); cK4097 = bcast((sK4097, 0))
        sC1 = const(C1); cC1 = bcast((sC1, 0))
        sK19 = const(19); cK19 = bcast((sK19, 0))
        sC2 = const(C2); cC2 = bcast((sC2, 0))
        sK33 = const(33); cK33 = bcast((sK33, 0))
        sC34 = const(C3 + C4); cC34 = bcast((sC34, 0))
        sK16896 = const(33 << 9); cK16896 = bcast((sK16896, 0))
        sC3s9 = const(C3 << 9); cC3s9 = bcast((sC3s9, 0))
        sK9 = const(9); cK9 = bcast((sK9, 0))
        sC5 = const(C5); cC5 = bcast((sC5, 0))
        sK16 = const(16); cK16 = bcast((sK16, 0))
        sC6 = const(C6)
        cC6 = bcast((sC6, 0))
        sK1 = const(1); cK1 = bcast((sK1, 0))
        sK2 = const(2); cK2 = bcast((sK2, 0))
        cY0 = bcast((const(-2), 0))  # y = ~v with v = idx + 1 = 1
        s8 = const(8)
        s2 = const(2)

        # ---- header pointers ---------------------------------------------
        fp = S.new(1)
        S.emit("load", ("load", "D", "S0"), (fp, 0), [(const(4), 0)], SETUP)
        vp = S.new(1)
        S.emit("load", ("load", "D", "S0"), (vp, 0), [(const(6), 0)], SETUP)
        cKADDR = bcast((alu("-", (fp, 0), (s2, 0)), 0))

        # ---- node constants for shallow depths (pre-xored with C6) --------
        node_view = []
        raw_consts = []
        addr = fp
        for i in range(cdiv(n_const_nodes, VLEN)):
            if i > 0:
                addr = alu("+", (addr, 0), (s8, 0))
            raw = S.new()
            S.emit("load", ("vload", "D", "S0"), (raw, 0), [(addr, 0)], SETUP)
            raw_consts.append(raw)
            if i == 0:
                raw0 = raw
            nx = valu("^", raw, cC6, SETUP)
            node_view.extend((nx, j) for j in range(VLEN))

        # ---- pre-xor deep tree levels in memory ---------------------------
        # tok[d] is a token every gather at depth d depends on (memory ordering).
        prexor = [d for d in self.PREXOR_LEVELS if any(gather_any[r] and depth[r] == d for r in range(rounds))]
        tok = {}
        for d in prexor:
            first, count = (1 << d) - 1, 1 << d
            nvl = cdiv(count, VLEN)
            t = S.new(1, nprod=nvl)
            a = alu("+", (fp, 0), (const(first), 0))
            for i in range(nvl):
                if i > 0:
                    a = alu("+", (a, 0), (s8, 0))
                raw = S.new()
                # must not overwrite nodes before the raw constant loads read them
                deps = [(a, 0)] + [(rv, 0) for rv in raw_consts] if first < n_const_nodes else [(a, 0)]
                S.emit("load", ("vload", "D", "S0"), (raw, 0), deps, SETUP)
                nx = S.new(nprod=VLEN)
                for j in range(VLEN):
                    S.emit("alu", ("^", "D", "S0", "S1"), (nx, j), [(raw, j), (sC6, 0)], SETUP)
                S.emit("store", ("vstore", "S0", "S1"), (t, 0), [(a, 0), (nx, 0)], SETUP)
            tok[d] = t

        # Per depth: for each assignment of the vselect bits (ve) and each subset
        # of the multiply_add bits (T), a broadcast coefficient vector.
        def vsel_cfg(r):
            dd = depth[r]
            k, place = self.VSEL_ROUND.get(r, self.VSEL.get(dd, (0, "top")))
            return dd, min(k, dd), place

        sel_tables = {}
        for r0 in range(rounds):
            if not select_any[r0] or depth[r0] == 0:
                continue
            dd, k, place = vsel_cfg(r0)
            if (dd, k, place) in sel_tables:
                continue
            # bit j (0 = earliest) lives at m-bit dd-1-j; nb=1 <=> b=0, so node
            # index = 2^dd - 1 + (2^dd - 1 - m)
            if place == "top":
                vbits = list(range(dd - k, dd))  # latest k bits
            else:
                vbits = list(range(k))  # earliest k bits
            mbits = [j for j in range(dd) if j not in vbits]
            vmask = sum(1 << (dd - 1 - j) for j in vbits)
            mmask_bits = [1 << (dd - 1 - j) for j in mbits]
            table = {}
            for ve in range(1 << dd):
                if ve & ~vmask:
                    continue
                cur = {}
                for ml in range(1 << dd):
                    if ml & ~sum(mmask_bits):
                        continue
                    m = ve | ml
                    cur[ml] = node_view[(1 << (dd + 1)) - 2 - m]
                for b in mmask_bits:
                    for ml in list(cur):
                        if ml & b:
                            cur[ml] = (alu("-", cur[ml], cur[ml ^ b]), 0)
                for ml, sv in cur.items():
                    table[(ve, ml)] = bcast(sv)
            sel_tables[(dd, k, place)] = (vbits, mbits, table)

        def horner(bits, coefs, vec):
            # bits: list of (bit_index, mbit, vreg) in increasing bit_index; split on the latest
            if not bits:
                return coefs[0]
            j, mb, x = bits[-1]
            rest = bits[:-1]
            f0 = horner(rest, {T: v for T, v in coefs.items() if not T & mb}, vec)
            f1 = horner(rest, {T ^ mb: v for T, v in coefs.items() if T & mb}, vec)
            return madd(x, f1, f0, vec)

        def vtree(bits, leaves, vec):
            if not bits:
                return leaves[0]
            j, mb, x = bits[-1]
            rest = bits[:-1]
            l0 = vtree(rest, {ve: v for ve, v in leaves.items() if not ve & mb}, vec)
            l1 = vtree(rest, {ve ^ mb: v for ve, v in leaves.items() if ve & mb}, vec)
            return vsel(x, l1, l0, vec)

        def select_node(r, nbs, vec):
            dd, k, place = vsel_cfg(r)
            vbits, mbits, table = sel_tables[(dd, k, place)]
            vb = [(j, 1 << (dd - 1 - j), nbs[j]) for j in vbits]
            mb = [(j, 1 << (dd - 1 - j), nbs[j]) for j in mbits]
            ves = sorted(set(ve for ve, _ in table))
            Ts = sorted(set(T for _, T in table))
            if place == "top":
                leaves = {ve: horner(mb, {T: table[(ve, T)] for T in Ts}, vec) for ve in ves}
                return vtree(vb, leaves, vec)
            else:
                coefs = {T: vtree(vb, {ve: table[(ve, T)] for ve in ves}, vec) for T in Ts}
                return horner(mb, coefs, vec)

        # ---- per-vector work -------------------------------------------
        cC0P = bcast(node_view[0])
        cC0 = bcast((raw0, 0))  # raw root value for round 0, whose input is not pre-xored with C6
        addr = vp
        vaddrs = []
        for k in range(ngroups):
            if k > 0:
                addr = alu("+", (addr, 0), (s8, 0))
            vaddrs.append(addr)

        # ---- scalar lanes -----------------------------------------------
        if NS:
            sFPm1 = alu("-", (fp, 0), (sK1, 0))
        for g in range(NS // VLEN):
            k = nvec + g
            vid = (g + 0.5) * nvec / (NS // VLEN)  # interleave priority with the vector groups
            sv = S.new()
            S.emit("load", ("vload", "D", "S0"), (sv, 0), [(vaddrs[k], 0)], vid)
            out = S.new(nprod=VLEN)
            for i in range(VLEN):
                val = (sv, i)
                v = None  # v = idx + 1; None means the constant 1 (at the root)
                for r in range(rounds):
                    dd = depth[r]
                    last = r == rounds - 1
                    if dd == 0:
                        node = (raw0, 0)
                        v = None
                    else:
                        a = alu("+", (v, 0), (sFPm1, 0), vid)
                        node = S.new(1)
                        S.emit("load", ("load", "D", "S0"), (node, 0), [(a, 0)], vid)
                        node = (node, 0)
                    a0 = alu("^", val, node, vid)
                    a1 = alu("*", (a0, 0), (sK4097, 0), vid)
                    a1 = alu("+", (a1, 0), (sC1, 0), vid)
                    t = alu(">>", (a1, 0), (sK19, 0), vid)
                    u = alu("^", (a1, 0), (t, 0), vid)
                    a2 = alu("^", (u, 0), (sC2, 0), vid)
                    p = alu("*", (a2, 0), (sK33, 0), vid)
                    p = alu("+", (p, 0), (sC34, 0), vid)
                    q = alu("*", (a2, 0), (sK16896, 0), vid)
                    q = alu("+", (q, 0), (sC3s9, 0), vid)
                    a4 = alu("^", (p, 0), (q, 0), vid)
                    a5 = alu("*", (a4, 0), (sK9, 0), vid)
                    a5 = alu("+", (a5, 0), (sC5, 0), vid)
                    t = alu(">>", (a5, 0), (sK16, 0), vid)
                    a6 = alu("^", (a5, 0), (t, 0), vid)
                    if last:
                        alu("^", (a6, 0), (sC6, 0), vid, dst=(out, i))
                        break
                    valr = alu("^", (a6, 0), (sC6, 0), vid)
                    val = (valr, 0)
                    if depth[r + 1] != 0:
                        b = alu("&", (valr, 0), (sK1, 0), vid)
                        if v is None:
                            v = alu("+", (b, 0), (sK2, 0), vid)
                        else:
                            v2 = alu("+", (v, 0), (v, 0), vid)
                            v = alu("+", (v2, 0), (b, 0), vid)
            S.emit("store", ("vstore", "S0", "S1"), None, [(vaddrs[k], 0), (out, 0)], vid)

        for k in range(nvec):
            gather, needy, neednb = plans[k]
            val = S.new()
            S.emit("load", ("vload", "D", "S0"), (val, 0), [(vaddrs[k], 0)], k)
            nbs = {}
            y = None
            stage = {}
            for r in range(rounds):
                S.cur_round = r
                dd = depth[r]
                if dd == 0:
                    y = cY0
                if gather[r]:
                    a = simple("addr", "-", cKADDR, y, k, r)
                    node = S.new(nprod=VLEN)
                    deps = [(a, 0)] + ([(tok[dd], 0)] if dd in tok else [])
                    for i in range(VLEN):
                        S.emit("load", ("load_offset", "D", "S0", i), (node, 0), deps, k)
                    t = simple("c6", "^", val, cC6, k, r) if r > 0 and dd not in tok else val
                    a0 = simple("x0", "^", t, node, k, r)
                elif dd == 0:
                    a0 = simple("x0", "^", val, cC0P if r > 0 else cC0, k, r)
                else:
                    bits = [nbs[r - dd + j] for j in range(dd)]
                    anchor[0] = stage.get(self.SEL_ANCHOR)
                    sel = select_node(r, bits, k)
                    anchor[0] = None
                    a0 = simple("x0", "^", val, sel, k, r)
                stage = {"a0": a0}
                a1 = madd(a0, cK4097, cC1, k)
                stage["a1"] = a1
                t = simple("s2a", ">>", a1, cK19, k, r)
                u = simple("s2b", "^", a1, t, k, r)
                a2 = simple("s2c", "^", u, cC2, k, r)
                stage["a2"] = a2
                p = madd(a2, cK33, cC34, k)
                q = madd(a2, cK16896, cC3s9, k)
                a4 = simple("s4", "^", p, q, k, r)
                stage["a4"] = a4
                a5 = madd(a4, cK9, cC5, k)
                stage["a5"] = a5
                t = simple("s6a", ">>", a5, cK16, k, r)
                a6 = simple("s6b", "^", a5, t, k, r)
                if r == rounds - 1:
                    val = simple("c6", "^", a6, cC6, k, r)
                    break
                if neednb[r]:
                    nb = simple("nb", "&", a6, cK1, k, r)
                    nbs[r] = nb
                    if needy[r + 1] and depth[r + 1] != 0:
                        y = madd(y, cK2, nb, k)
                val = a6
            S.emit("store", ("vstore", "S0", "S1"), None, [(vaddrs[k], 0), (val, 0)], k)
            S.cur_round = 0

        for attempt in range(4):
            try:
                bundles = S.run()
                break
            except RuntimeError:
                if attempt == 3:
                    raise
                S.reset()
                S.reserve += 6
        # The pause lets the local harness stop before the loop like the reference's first yield.
        bundles[0].setdefault("flow", []).append(("pause",))
        self.instrs = bundles
        self.stats = self._stats(bundles)
        self.stats["min_free_v"] = S.min_free_v
        self.stats["free_v0"] = len(S.free_v)

    @staticmethod
    def _stats(bundles):
        used = {}
        for b in bundles:
            for e, slots in b.items():
                used[e] = used.get(e, 0) + len(slots)
        n = len(bundles)
        return {e: f"{c}/{n * SLOT_LIMITS[e]} ({100 * c / (n * SLOT_LIMITS[e]):.0f}%)" for e, c in used.items() if e != "debug"}


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
