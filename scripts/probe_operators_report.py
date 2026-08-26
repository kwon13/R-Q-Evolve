#!/usr/bin/env python
"""Score the operator probe: validity, novelty, diversity -- one row per operator."""
from __future__ import annotations
import argparse, ast, difflib, json, math, statistics, collections, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rq_evolve.code_utils import lint_generator_source   # noqa: E402
from rq_evolve.program import ProblemProgram             # noqa: E402

class _Sk(ast.NodeVisitor):
    def __init__(self): self.seq=[]
    def generic_visit(self,n): self.seq.append(type(n).__name__); super().generic_visit(n)
def skel(src):
    try: t=ast.parse(src)
    except Exception: return None
    v=_Sk(); v.visit(t); return tuple(v.seq)
def sim(a,b): return difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rows", default="rq_output/probe_operators.jsonl")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_4gpu/rq_archive")
    ap.add_argument("--valid-out", default="rq_output/probe_operators_valid.jsonl")
    args=ap.parse_args()
    rows=[json.loads(l) for l in (ROOT/args.rows).read_text().splitlines() if l.strip()]
    snaps=sorted((ROOT/args.archive).glob("archive_iter*.json"),
                 key=lambda p:int("".join(c for c in p.name if c.isdigit())))
    ch=json.loads(snaps[-1].read_text())["champions"]; ch=list(ch.values() if isinstance(ch,dict) else ch)
    arch=[s for s in (skel(c["source_code"]) for c in ch) if s]

    ops=[]; seen=set()
    for r in rows:
        if r["op"] not in seen: seen.add(r["op"]); ops.append(r["op"])
    hdr=(f"{'operator':<15}{'s1':>4}{'s2':>4}{'exec':>5}{'lint':>5}"
         f"{'구조유사':>9}{'텍스트':>8}{'골격동일':>9}{'아카이브최근접':>13}"
         f"{'고유골격':>10}{'스킬종류':>9}")
    print(hdr); print("-"*len(hdr)+"-"*10)
    valid_rows=[]
    for op in ops:
        g=[r for r in rows if r["op"]==op]
        s1=sum(1 for r in g if r.get("stage1_ok")); s2=sum(1 for r in g if r.get("stage2_ok"))
        ss=[]; ts=[]; an=[]; sk_ok=[]; nexec=0; nlint=0; skills=collections.Counter()
        for r in g:
            c=r.get("code")
            if not c: continue
            s=skel(c)
            if s is None: continue
            p=skel(r["parent_src"])
            if p: ss.append(sim(s,p)); ts.append(sim(c,r["parent_src"]))
            an.append(max((sim(s,a) for a in arch), default=0.0))
            sk_ok.append(s); skills[r.get("plan_skill")]+=1
            if not lint_generator_source(c): nlint+=1
            try:
                prog=ProblemProgram(source_code=c)
                insts=[prog.execute(seed=z) for z in range(3)]
                if all(i is not None for i in insts) and len({i.problem for i in insts})>=2:
                    nexec+=1
                    r["_valid"]=True; valid_rows.append(r)
            except Exception: pass
        n=len(sk_ok) or 1
        uniq=len(set(sk_ok))
        ident=sum(1 for x in ss if x>=0.999)
        med=lambda v: statistics.median(v) if v else float("nan")
        print(f"{op:<15}{s1:>4}{s2:>4}{nexec:>5}{nlint:>5}"
              f"{med(ss):>9.3f}{med(ts):>8.3f}{100*ident/n:>8.0f}%{med(an):>13.3f}"
              f"{uniq:>6}/{n:<3}{len(skills):>9}")
    (ROOT/args.valid_out).write_text("\n".join(json.dumps(r) for r in valid_rows), encoding="utf-8")
    print(f"\n구조유사/텍스트 = 부모와의 중앙값 | 골격동일 = AST skeleton 완전 일치 비율")
    print(f"아카이브최근접 = 아카이브 어느 프로그램과든 가장 가까운 구조 유사도 (낮을수록 새로움)")
    print(f"\n실행 가능한 자식 {len(valid_rows)}개 -> {ROOT/args.valid_out}")
    return 0
if __name__=="__main__": raise SystemExit(main())
