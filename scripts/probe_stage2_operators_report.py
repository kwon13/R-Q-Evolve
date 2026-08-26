#!/usr/bin/env python
"""Score the stage-2 operator probe. Every headline twice: all parseable, and EXECUTABLE only (D4).

Calibration for the skeleton-similarity column, measured on this repo:
  hand-written seed_programs, pairwise            0.397   <- what real diversity reads as
  champions with different coarse shapes          0.648
  champions with the same coarse shape            0.756
  parent -> accepted child (the failure)          0.996
An arm that reaches 0.65 has produced a genuinely different algorithm. 0.40 is
the floor, not 0.
"""
from __future__ import annotations
import argparse, ast, difflib, importlib.util, json, re, signal, statistics, sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sf", ROOT / "src" / "rq_evolve" / "structural_fingerprint.py")
SF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SF)


class _Sk(ast.NodeVisitor):
    def __init__(self): self.seq = []
    def generic_visit(self, n): self.seq.append(type(n).__name__); super().generic_visit(n)


def skel(src):
    try: t = ast.parse(src)
    except Exception: return None
    v = _Sk(); v.visit(t); return tuple(v.seq)


def sim(a, b): return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_TO()))


def run3(src):
    g = {}
    try:
        signal.alarm(6)
        exec(compile(src, "<c>", "exec"), g)
        out = [g["generate"](z) for z in range(3)]
        signal.alarm(0)
    except BaseException:
        signal.alarm(0); return None
    if len({o[0] for o in out}) < 2: return None
    return [(o[0], str(o[1])) for o in out]


def leaks(pairs):
    for p, a in pairs:
        a = a.strip()
        if not a or re.search(r"(?<![\d.])" + re.escape(a) + r"(?![\d.])", p) is None:
            return False
    return True


def answer_line(src):
    for l in src.splitlines():
        if l.strip().startswith("answer ="): return l.strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="rq_output/probe_stage2_operators.jsonl")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / args.rows).read_text().splitlines() if l.strip()]
    arms = list(dict.fromkeys(r["arm"] for r in rows))

    hdr = (f"{'arm':<23}{'n':>4}{'exec%':>7}{'skelP(all)':>11}{'skelP(exec)':>12}"
           f"{'bare%(ex)':>10}{'leak%(ex)':>10}{'ansLineCopy':>12}{'shapes':>8}{'newShape%':>10}")
    print(hdr); print("-" * len(hdr))
    ctrl_by_parent = {}
    for r in rows:
        if r["arm"] == "parent_source" and r.get("code"):
            ctrl_by_parent[r["parent_i"]] = r["code"]
    for arm in arms:
        g = [r for r in rows if r["arm"] == arm and r.get("code")]
        ss, sse, nex, nb, nlk, nal, shapes, newshape = [], [], 0, 0, 0, 0, set(), 0
        for r in g:
            c = r["code"]; sc = skel(c); sp = skel(r["parent_src"])
            pairs = run3(c); ok = pairs is not None
            if ok: nex += 1
            if sc and sp:
                ss.append(sim(sc, sp))
                if ok: sse.append(sim(sc, sp))
            fp = SF.structural_fingerprint(c); pfp = SF.structural_fingerprint(r["parent_src"])
            if fp and ok:
                shapes.add(SF.shape_key(fp))
                if "BARE COPY" in fp["answer_shape"]: nb += 1
                if pfp and SF.shape_key(fp) != SF.shape_key(pfp): newshape += 1
            if ok and leaks(pairs): nlk += 1
            if answer_line(c) and answer_line(c) == answer_line(r["parent_src"]): nal += 1
        n = len(g) or 1; e = nex or 1
        m = lambda v: statistics.median(v) if v else float("nan")
        print(f"{arm:<23}{len(g):>4}{100*nex/n:>6.0f}%{m(ss):>11.3f}{m(sse):>12.3f}"
              f"{100*nb/e:>9.0f}%{100*nlk/e:>9.0f}%{100*nal/n:>11.0f}%{len(shapes):>8}{100*newshape/e:>9.0f}%")

    # Paired: same parent, same child family, control vs arm.
    print("\nPAIRED vs parent_source (same parent, identical child family):")
    for arm in arms:
        if arm == "parent_source": continue
        d = []
        for r in rows:
            if r["arm"] != arm or not r.get("code"): continue
            c0 = ctrl_by_parent.get(r["parent_i"])
            if not c0: continue
            a, b = skel(r["code"]), skel(c0); p = skel(r["parent_src"])
            if a and b and p: d.append(sim(a, p) - sim(b, p))
        if d:
            wins = sum(1 for x in d if x < 0)
            print(f"  {arm:<23} median Δ skeleton-sim {statistics.median(d):+.3f}   "
                  f"more novel than control on {wins}/{len(d)} paired parents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
