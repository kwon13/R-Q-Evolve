"""Figure: does the Evolver's two-operator mutation expand the problem space?

Panel A: t-SNE map of TF-IDF-embedded problem texts (all programs ever archived),
         colored by originating operator, lineage edges parent->child.
Panel B: cumulative distinct concept_types in the archive, by introducing operator.
Panel C: p-hat of accepted (inserted) children over iterations, by operator.
"""
import argparse, json, re, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from collections import Counter, defaultdict
from viz_common import family_history, load_snapshots, operator_of
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
except ImportError:
    TfidfVectorizer = None
    TSNE = None

parser = argparse.ArgumentParser()
parser.add_argument('--rq-output', required=True)
parser.add_argument('--analysis-output', required=True)
parser.add_argument('--champions-with-texts', required=True)
parser.add_argument('--peak-global-step', type=int, default=None)
parser.add_argument('--peak-outer-iteration', type=int, default=None)
parser.add_argument('--max-outer-iteration', type=int, default=None)
args = parser.parse_args()
TEXTS_PATH = args.champions_with_texts
BASE = f'{args.rq_output}/rq_archive'
OUT = args.analysis_output
PEAK_GLOBAL_STEP = args.peak_global_step
PEAK_OUTER_ITER = args.peak_outer_iteration

C_OP = {'seed': '#8a8984', 'in_depth': '#2a78d6', 'in_breadth': '#eb6834'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
RUN_NAME = args.rq_output.rstrip('/').split('/')[-1]

progs = json.loads(Path(TEXTS_PATH).read_text(encoding="utf-8"))
progs = [p for p in progs if any(str(t).strip() for t in p.get("problem_texts", []))]
if not progs:
    raise ValueError(f"no successfully generated problem texts in {TEXTS_PATH}")
for p in progs:
    p["op"] = operator_of(p)
by_id = {p['program_id']: p for p in progs}

# ---------------- embedding ----------------
docs = []
for p in progs:
    t = ' '.join(p['problem_texts'])
    t = re.sub(r'\d+', ' ', t)              # drop seed-dependent numerals
    docs.append(t)
if TfidfVectorizer is not None and len(progs) >= 4:
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, max_features=4000,
                          sublinear_tf=True)
    X = vec.fit_transform(docs)
    perplexity = min(12, max(2, (len(progs) - 1) // 3))
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=0, init='pca',
               metric='cosine', learning_rate='auto').fit_transform(X.toarray())
    embedding_name = 't-SNE of TF-IDF'
else:
    # Dependency-free fallback for the training environment: build a compact
    # unigram/bigram TF-IDF matrix, then use its first two centered SVD axes.
    tokenized = []
    document_frequency = Counter()
    for doc in docs:
        words = re.findall(r"[a-zA-Z_]{2,}", doc.lower())
        terms = words + [f"{a}__{b}" for a, b in zip(words, words[1:])]
        counts = Counter(terms)
        tokenized.append(counts)
        document_frequency.update(counts)
    vocab = [
        term for term, _ in sorted(
            document_frequency.items(), key=lambda item: (-item[1], item[0])
        )[:4000]
    ]
    index = {term: i for i, term in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    for row, counts in enumerate(tokenized):
        for term, count in counts.items():
            if term in index:
                X[row, index[term]] = 1.0 + math.log(count)
    if X.shape[1]:
        idf = np.array([
            math.log((1.0 + len(docs)) / (1.0 + document_frequency[term])) + 1.0
            for term in vocab
        ])
        X *= idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X /= np.where(norms == 0, 1.0, norms)
    centered = X - X.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    emb = u[:, :2] * s[:2]
    if emb.shape[1] < 2:
        emb = np.pad(emb, ((0, 0), (0, 2 - emb.shape[1])))
    embedding_name = 'SVD of TF-IDF'
for p, (x, y) in zip(progs, emb):
    p['x'], p['y'] = float(x), float(y)

# ---------------- panel B data ----------------
snapshots = load_snapshots(Path(BASE), args.max_outer_iteration)
_, first_type_raw = family_history(
    snapshots, Path(BASE) / "evolution_log.jsonl"
)
first_type = {typ: (op, iteration) for typ, (iteration, op) in first_type_raw.items()}
iters = [iteration for iteration, _ in snapshots]
cum = {op: [] for op in C_OP}
for it in iters:
    for op in C_OP:
        cum[op].append(sum(1 for o, i in first_type.values() if o == op and i <= it))

# ---------------- panel C data ----------------
ins = []                  # (iter, op, p_hat)
for line in Path(BASE, "evolution_log.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    if args.max_outer_iteration is not None and rec.get('iteration', -1) > args.max_outer_iteration:
        continue
    for r in rec.get('reports', []):
        if r.get('status') == 'inserted' and r.get("s_hat", r.get("p_hat")) is not None:
            p_hat = float(r.get("s_hat", r["p_hat"]))
            if math.isfinite(p_hat):
                ins.append((int(rec['iteration']), r['op'], p_hat))

# ---------------- figure ----------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'text.color': INK,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
fig = plt.figure(figsize=(13.5, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], hspace=0.42, wspace=0.18,
                      left=0.05, right=0.98, top=0.90, bottom=0.09)
axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 1])

# ---- Panel A: map ----
axA.grid(False)
axA.set_xticks([]); axA.set_yticks([])
for s in axA.spines.values():
    s.set_visible(True); s.set_color(GRID)

# lineage edges
for p in progs:
    par = by_id.get(p.get('parent_id'))
    if not par:
        continue
    axA.annotate('', xy=(p['x'], p['y']), xytext=(par['x'], par['y']),
                 arrowprops=dict(arrowstyle='-|>', color=C_OP[p['op']], alpha=0.45,
                                 lw=1.0, shrinkA=4, shrinkB=4,
                                 connectionstyle='arc3,rad=0.12'))
# points
for op, marker, size in [('seed', 's', 62), ('in_depth', 'o', 55), ('in_breadth', 'o', 55)]:
    pts = [p for p in progs if p['op'] == op]
    axA.scatter([p['x'] for p in pts], [p['y'] for p in pts],
                s=size, marker=marker, c=C_OP[op], edgecolors='white', linewidths=1.4,
                zorder=3, label={'seed': 'Seed program', 'in_depth': 'in-depth child',
                                 'in_breadth': 'in-breadth child'}[op])
# Concept-group labels at cluster medians.  SVD can place several group medians
# nearly on top of each other, so fan the labels around their anchors.
group_names = sorted({p['group'] for p in progs if p['group']})
label_offsets = [(-62, 38), (0, 55), (62, 38), (-62, -40), (0, -55), (62, -40)]
x_min, x_max = float(emb[:, 0].min()), float(emb[:, 0].max())
y_min, y_max = float(emb[:, 1].min()), float(emb[:, 1].max())
x_span, y_span = max(x_max - x_min, 1e-9), max(y_max - y_min, 1e-9)
for label_index, g in enumerate(group_names):
    pts = [p for p in progs if p['group'] == g]
    gx = float(np.median([p['x'] for p in pts]))
    gy = float(np.median([p['y'] for p in pts]))
    dx, dy = label_offsets[label_index % len(label_offsets)]
    if gx > x_min + .82 * x_span:
        dx = -abs(dx)
    elif gx < x_min + .18 * x_span:
        dx = abs(dx)
    if gy > y_min + .82 * y_span:
        dy = -abs(dy)
    elif gy < y_min + .18 * y_span:
        dy = abs(dy)
    offset = (dx, dy)
    axA.annotate(
        g.replace('_', ' '), xy=(gx, gy), xytext=offset,
        textcoords='offset points', fontsize=8.5, style='italic', color=INK2,
        ha='center', va='center', zorder=5,
        bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor=GRID, alpha=.9),
        arrowprops=dict(arrowstyle='-', color='#aaa7a1', lw=.8),
    )

axA.legend(loc='upper left', frameon=False, fontsize=9, handletextpad=0.3)
axA.set_title(f'A · Problem space ({embedding_name} problem text), all archived programs',
              fontsize=11, loc='left', color=INK)
if PEAK_GLOBAL_STEP is not None:
    peak_label = f'performance peak: global step {PEAK_GLOBAL_STEP}'
    peak_label += f'\nouter iteration {PEAK_OUTER_ITER}' if PEAK_OUTER_ITER is not None else '\nouter iteration unavailable'
    axA.text(0.99, 0.02, peak_label, transform=axA.transAxes, ha='right', va='bottom', fontsize=8.5, color=INK2,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GRID, alpha=.9))

# ---- Panel B: cumulative distinct concept types ----
labels = {'seed': 'seed', 'in_depth': 'in-depth', 'in_breadth': 'in-breadth'}
for op in ['in_breadth', 'in_depth', 'seed']:
    axB.plot(iters, cum[op], color=C_OP[op], lw=2, solid_capstyle='round')
    axB.text(iters[-1] + 0.8, cum[op][-1], f"{labels[op]} ({cum[op][-1]})",
             color=C_OP[op], fontsize=9, va='center')
axB.set_xlim(0, max(iters) + 5)
if PEAK_OUTER_ITER is not None:
    axB.axvline(PEAK_OUTER_ITER, color=INK, ls='--', lw=1.1, alpha=.8)
    axB.text(PEAK_OUTER_ITER + .7, axB.get_ylim()[1] * .95,
             f'global step {PEAK_GLOBAL_STEP}\nouter iter {PEAK_OUTER_ITER}', fontsize=8,
             color=INK2, va='top')
elif PEAK_GLOBAL_STEP is not None:
    axB.text(.99, .96, f'peak: global step {PEAK_GLOBAL_STEP}\nouter iteration unavailable',
             transform=axB.transAxes, ha='right', va='top', fontsize=8, color=INK2)
axB.set_title('B · Distinct problem families first observed, by origin',
              fontsize=11, loc='left', color=INK)
axB.set_xlabel('outer iteration', fontsize=9)
axB.set_ylabel('cumulative distinct families', fontsize=9)
for s in ['top', 'right']:
    axB.spines[s].set_visible(False)

# ---- Panel C: p-hat of accepted children ----
axC.axhspan(0.2, 0.8, color='#f3f2ee', zorder=0)
in_band = sum(1 for _, _, ph in ins if 0.2 <= ph <= 0.8) / len(ins) if ins else 0.0
axC.text(
    .02, .82,
    f'learnable band 0.2–0.8  ({in_band:.0%} of accepted)'
    if ins else 'no accepted children logged',
    transform=axC.transAxes, fontsize=8, color=INK2, va='bottom', ha='left',
)
for op in ['in_depth', 'in_breadth']:
    xs = [i for i, o, _ in ins if o == op]
    ys = [ph for _, o, ph in ins if o == op]
    axC.scatter(xs, ys, s=26, c=C_OP[op], edgecolors='white', linewidths=0.8, zorder=3)
axC.set_ylim(-0.05, 1.05)
axC.set_xlim(0, max([i for i, _, _ in ins], default=1) + 3)
if PEAK_OUTER_ITER is not None:
    axC.axvline(PEAK_OUTER_ITER, color=INK, ls='--', lw=1.1, alpha=.8)
axC.set_title('C · Solver pass rate $\\hat{p}$ of accepted children at insertion',
              fontsize=11, loc='left', color=INK)
axC.set_xlabel('outer iteration', fontsize=9)
axC.set_ylabel('$\\hat{p}$', fontsize=9)
for s in ['top', 'right']:
    axC.spines[s].set_visible(False)

fig.suptitle(f'Evolver mutation: archived problem space and family introduction  ·  {RUN_NAME}',
             fontsize=12.5, x=0.05, ha='left', color=INK, fontweight='bold')
fig.savefig(f'{OUT}/problem_space_expansion.png', dpi=180)
fig.savefig(f'{OUT}/problem_space_expansion.pdf')
print('saved', f'{OUT}/problem_space_expansion.png')
