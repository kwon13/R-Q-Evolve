#!/usr/bin/env python3
"""Problem-grammar profile: how close does each arm's CHILD FAMILY come to the
competition benchmarks it is supposed to train for?

Axes are the ones that separate an AIME statement from a formula lookup:
  words        statement length
  numbers      numeric givens in the rendered statement
  params       free parameters in the generator template (1 = one knob)
  how-many     share opening with "How many" -- AIME never does
  composite    share asking for a combination of the unknowns rather than a given
  normalizer   share closing in the contest idiom that forces a small integer
"""
import argparse, json, os, re, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_TEX = re.compile(r'\\[a-zA-Z]+|[\\${}]')
_NUM = re.compile(r'(?<![A-Za-z_])\d+')
_HOWMANY = re.compile(r'\bhow many\b', re.I)
_ASK = re.compile(r'\b(find|determine|compute|what is|how many|calculate)\b(.{0,90})', re.I)
_COMPOSITE = re.compile(r'[a-zA-Z]\s*[\^*+]\s*\d?\s*[a-zA-Z]|\b\w+\s*\+\s*\w+\b')
_NORM = re.compile(r'relatively prime|\bm\s*\+\s*n\b|\bp\s*\+\s*q\b|remainder when .{0,40}divided by'
                   r'|can be (written|expressed)|sum of the digits', re.I)
_PARAM = re.compile(r'\{([^}]{1,40})\}')
_FORMAT_INSTR = re.compile(r'state only the integer|answer with (only )?an? integer|output only', re.I)


def profile(text, template=None):
    t = ' '.join(str(text or '').split())
    plain = _TEX.sub(' ', t)
    ask = _ASK.search(plain)
    return dict(
        words=len(plain.split()),
        numbers=len(_NUM.findall(plain)),
        params=len(set(_PARAM.findall(template if template is not None else t))),
        givens=len(_NUM.findall(plain)) + len(set(_PARAM.findall(template if template is not None else t))),
        howmany=bool(_HOWMANY.search(plain)),
        composite=bool(_COMPOSITE.search(ask.group(2))) if ask else False,
        normalizer=bool(_NORM.search(plain)),
        format_instr=bool(_FORMAT_INSTR.search(plain)),
    )


def summarize(name, profs):
    n = len(profs)
    if not n:
        return
    med = lambda k: st.median(p[k] for p in profs)
    sh = lambda k: 100 * sum(p[k] for p in profs) / n
    print(f'{name:26s}{n:5d}{med("words"):8.0f}{med("numbers"):9.0f}{med("params"):8.1f}'
          f'{sh("howmany"):10.0f}%{sh("composite"):11.0f}%{sh("normalizer"):12.0f}%{sh("format_instr"):9.0f}%')


HEADER = (f'{"source":26s}{"n":>5s}{"words":>8s}{"givens":>9s}{"params":>8s}'
          f'{"how-many":>11s}{"composite":>12s}{"normalizer":>13s}{"fmt-instr":>10s}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="rq_output/gate_experiment")
    ap.add_argument("--tag", default="")
    ap.add_argument("--arms", default="baseline,target_cell,target_rotate,target_rotate_full,target_grammar")
    ap.add_argument("--reference", action="store_true", help="also print benchmark + seed reference rows")
    args = ap.parse_args()

    print(HEADER)
    print('-' * len(HEADER))
    if args.reference:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        from rq_evolve.math_eval import _load_benchmark_rows
        from rq_evolve.program import ProblemProgram
        for b in ("aime24", "amc23", "math500"):
            try:
                rows = _load_benchmark_rows(b, inflate=False)
            except Exception as e:
                print(f'  (skipped {b}: {e})'); continue
            summarize(f'BENCH {b}', [profile(r["question"], template="") for r in rows])
        seeds = []
        for f in sorted(os.listdir("seed_programs")):
            src = open("seed_programs/" + f).read()
            inst = ProblemProgram(source_code=src, program_id=f).execute(seed=0)
            if inst:
                from rq_evolve.code_utils import extract_problem_template
                seeds.append(profile(inst.problem,
                                     template=extract_problem_template(src) or ""))
        summarize("SEED programs", seeds)
        print('-' * len(HEADER))

    for arm in args.arms.split(","):
        p = os.path.join(args.dir, f"raw_{arm}{args.tag}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        profs = [profile(r["child_family"], template=r["child_family"])
                 for r in rows if r["stage1_parsed"] and r["child_family"]]
        summarize(arm + args.tag, profs)


if __name__ == "__main__":
    main()
