"""Execute each archived champion's generate(seed) to collect problem texts."""
import json, multiprocessing as mp, sys

SCRATCH = '/tmp/claude-1024/-data1-yhoon113/13ac5038-7836-43e5-84d3-108dd436a1fa/scratchpad'
SEEDS = [0, 1, 2]


def run_one(src):
    ns = {}
    exec(src, ns)
    out = []
    for s in SEEDS:
        try:
            pt, ans = ns['generate'](s)
            out.append(str(pt))
        except Exception:
            pass
    return out


def worker(src, q):
    try:
        q.put(run_one(src))
    except Exception as e:
        q.put({'error': repr(e)[:200]})


def main():
    progs = json.load(open(f'{SCRATCH}/champions.json'))
    for p in progs:
        q = mp.Queue()
        proc = mp.Process(target=worker, args=(p['source_code'], q))
        proc.start()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate(); proc.join()
            p['problem_texts'] = []
            p['gen_error'] = 'timeout'
        else:
            r = q.get() if not q.empty() else []
            if isinstance(r, dict):
                p['problem_texts'] = []
                p['gen_error'] = r['error']
            else:
                p['problem_texts'] = r
    json.dump(progs, open(f'{SCRATCH}/champions_with_texts.json', 'w'))
    ok = sum(1 for p in progs if p.get('problem_texts'))
    print(f'{ok}/{len(progs)} programs produced texts')
    for p in progs:
        if not p.get('problem_texts'):
            print('FAILED:', p['program_id'], p.get('gen_error'))


if __name__ == '__main__':
    main()
