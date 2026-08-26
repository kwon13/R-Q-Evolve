"""Adversarial-review corrections to the v2 stage-1 operator design.

Every number below was recomputed from rq_output/probe_operators.jsonl (300 rows)
with /data1/yhoon113/miniforge3/envs/vllm/bin/python. The v2 proposal's per-arm
figures all reproduce exactly; these are the places its DESIGN does not follow
from them.

-----------------------------------------------------------------------------
F1. `parameter` is UNSATISFIABLE in stage 1's channel. The neutral prompt
    cannot fix it, so the "instrument check" fails for a reason unrelated to D1.

    stage1() shows the parent as extract_problem_template(...) -- the f-string
    with {braces}. The "numeric ranges its parameters are drawn from" live in
    rng.randint(1, 10) in the SOURCE, which stage 1 never sees. Stage 1's output
    is prose re-parameterised with {names}, so there is no slot in which to
    write "the range is now 1..50" either. The operator asks for a change the
    arm can neither observe nor express.

    Measured consequence (stage-1 fidelity = SequenceMatcher of CHILD FAMILY to
    the parent problem TEMPLATE, medians over stage1_ok rows):
        scale      0.851   <- low-strength AND obeyed, under the SAME shipped head
        parameter  0.630   <- low-strength and NOT obeyed
    Both arms faced the identical "Changing only parameters ... does not count"
    head. `scale` resisted it; `parameter` did not. The head is therefore not
    what distinguishes them, and the parameter/scale inversion is not "the
    entire evidentiary basis for D1".

    Few-shot copying, same rows (CHILD FAMILY within 0.75 normalised similarity
    of a canned EXAMPLE's child family):
        parameter  4/29 = 14%  (EXAMPLE 8 twice)   scale 1/29 = 3%
    i.e. a large share of parameter's "novelty" is the EXAMPLE 8 (Mantel)
    attractor that prompts.py:468-478 documents at 12-16% -- which is D2, not
    D1. v2 changes D1 and D2 at once, so its PRIMARY READOUT cannot attribute
    the result to either.

FIX: keep `scale` as the low-strength control (it is the arm that empirically
behaves like one), and either repair `parameter` so it is expressible or replace
it with a true null. Both corrected texts below.
-----------------------------------------------------------------------------
"""

# --- F1 fix, option A: a null arm that IS expressible in stage 1's channel ----
NULL_ARM = """
Reproduce the PARENT FAMILY as your CHILD FAMILY, word for word, with the same
parameter names in the same places. Change nothing at all. This is a control:
an exact copy is the correct answer, not a failure to follow instructions.
"""

# --- F1 fix, option B: keep a parameter arm, but make it observable ----------
# Requires stage1() to also show the parent's draw statements, which it can lift
# straight out of the source; the family text alone does not contain them.
PARAMETER_FIXED = """
The parent's parameter draws are listed under PARENT PARAMETER RANGES. Keep the
parent's statement structure exactly and keep every parameter name. State the
new range for each parameter on its own line under CHANGE MADE, in the form
`name: lo..hi`. Change nothing else: the objects, the quantity asked for, and
the solution route stay as they are.
"""
# and in stage1(), after the reference instance:
#   import re
#   draws = "\n".join(l.strip() for l in parent.source_code.splitlines()
#                     if re.search(r"=\s*rng\.(randint|randrange|choice|sample)", l))
#   if op in ("parameter",) and draws:
#       body += f"\nPARENT PARAMETER RANGES\n\n{draws}\n"

# -----------------------------------------------------------------------------
# F2. The instrument check must not be read off a STAGE-2 artifact.
#
# Code-skeleton similarity is written by stage 2, which is byte-identical in
# every arm (D5, unfixed). Pooling all 300 rows, code-skeleton similarity to the
# parent as a function of stage-1 fidelity:
#
#   stage-1 fidelity   n    median code sim   max     >=0.95
#   [0.00,0.40)       75         0.609        0.965    1
#   [0.40,0.60)       50         0.721        0.974    7
#   [0.60,0.80)       44         0.819        0.985   12
#   [0.80,0.90)       31         0.955        0.974   16
#   [0.90,1.00]       21         0.963        0.976   16
#
#   Pearson r(stage-1 fidelity, code skeleton sim) = 0.630
#
# So stage 1 explains ~40% of the variance in the outcome the probe scores, and
# the STAGE-2 CEILING is 0.963 median / 0.976 max even when stage 1 hands over a
# near-verbatim copy. The v2 rule "parameter must climb to >=0.95 or every other
# number in the run is unreadable" is a threshold 0.013 under a hard ceiling,
# applied to a quantity stage 1 only partly controls, with a pre-committed
# consequence of discarding the whole run.
#
# FIX: the instrument check is stage-1 FIDELITY (prose vs the parent template),
# target ~1.00 for NULL_ARM. Report code-skeleton similarity as a SECONDARY,
# stage-2-contaminated readout, and never use it to invalidate the run.

def stage1_fidelity(child_family, parent_src):
    """The instrument check. Independent of stage 2."""
    import re, difflib
    from rq_evolve.code_utils import extract_problem_template
    pt = extract_problem_template(parent_src) or ""
    n = lambda s: re.sub(r"\s+", " ", s.strip())
    return difflib.SequenceMatcher(None, n(pt), n(child_family), autojunk=False).ratio()


# -----------------------------------------------------------------------------
# F3. "Gate yield" is not the pipeline. Recomputed against the gates that were
# actually fixed today (exec + lint + answer_is_bare_draw +
# answer_leaks_in_every_instance + AST skeleton 0.90 + near-duplicate template
# 0.85), over the same 300 rows and the same 48-champion snapshot:
#
#   op            attempts  exec  bare  leak  AST.90  ACCEPT   v2's "gate yield"
#   feedback         30      14     3     0     7       7            7
#   local_diff       30      10     2     0     7       5            7
#   structural       30      10     2     1     6       5            7
#   adaptive         30      13     0     0     7       5            7
#   donor            30       9     0     0     8       4            8   <-- 1st -> mid
#   full_rewrite     30      12     3     1     4       4            5
#   parameter        30      10     0     0     6       4            6
#   simplify         30      14     4     0     5       3            5
#   directed         30      15     3     2     2       2            3
#   scale            30      14     3     1     4       2            5
#   TOTAL           300     121    20     5    56      41           60
#
# The near-duplicate TEMPLATE gate (0.85) alone removes 15 of the 56, and it hits
# donor hardest. Under the real pipeline donor is 4/30 = 0.133, not 0.267, and
# feedback leads at 7/30. Every "donor is the highest-yield arm / the only
# mechanism that can break the lineage" claim rests on the truncated metric.
#
# FIX: score acceptance with the whole gate stack, and freeze + record the
# archive snapshot path (v2 risk 4 is right about that part).

# -----------------------------------------------------------------------------
# F4. The power argument is inverted by its own cut.
#
# v2's n=81 comes from a 0.27-vs-0.10 contrast. The 0.10 arm is `directed`,
# which v2 CUT. Among the five arms v2 KEEPS, v1 gate yields are
# donor .267, local_diff .233, structural .233, feedback .233, parameter .200.
# n per arm at 80% power, alpha=.05, for every contrast v2 can still make:
#
#   donor vs local_diff / structural / feedback   n = 2648   (three exact ties)
#   donor vs parameter                            n =  631
#   local_diff / structural / feedback vs parameter  n = 2397
#   donor vs directed  (the CUT arm)              n =   84
#
# Actual power at n=75, unpaired: 0.27 vs 0.10 -> 0.77 (not 0.80; 75 < 81).
#                                 0.27 vs 0.17 -> 0.31
#                                 0.27 vs 0.23 -> 0.08
# With 5 arms = 10 pairwise contrasts, Bonferroni alpha=.005:
#                                 0.27 vs 0.10 at n=75 -> power 0.45; n needed 138.
#
# v2's SECONDARY readout is "donor vs structural at p<0.05". v1 measured that
# pair 8/30 vs 7/30, Fisher p = 1.000, required n = 2648/arm.
#
# And the pairing does not rescue it. Per-parent heterogeneity of the gate
# outcome across the 39 parents with >=4 attempts:
#   chi2 = 33.4, df = 38, p = 0.68; observed var 0.0232 vs binomial 0.0255
#   => parent-attributable variance share ~ 0.000; effective-n multiplier 1.00x
# Pairing on the parent buys NOTHING for the proportion metric v2 powers on.
# (It does help for EXEC rate: chi2 = 62.6, df = 38, p = 0.007, parent share
# ~0.41 -- so keep the pairing, just do not claim power from it.)
#
# FIX: stop powering on between-operator gate yield. Power on the ONE contrast
# the data supports -- shipped head vs neutral head on the SAME operator, which
# is a within-pair contrast on a large expected effect -- and report all
# operator orderings as descriptive with CIs.

# -----------------------------------------------------------------------------
# F5. (a)-vs-(b) is a false dichotomy, and (a) deletes the control.
#
# v2 removes the shipped head from EVERY arm, so no column of the experiment is
# production. But D1 is a HYPOTHESIS about the head, and the head is a factor you
# can cross rather than delete. Minimal factorial that answers D1 directly:
#
#   HEAD in {shipped, neutral}  x  OP in {NULL_ARM, structural}     = 4 cells
#   plus  HEAD=neutral          x  OP in {scale, donor, feedback}   = 3 cells
#
# 7 cells x n=75 = 525 rows. D1's effect is then the shipped-vs-neutral delta on
# NULL_ARM -- a difference the v1 data predicts is large (fidelity 0.63 -> ~1.0),
# which is the only contrast in this whole design that n=75 can actually resolve.
# Cost check, measured on the live step160 server at port 8701 while the 4B run
# is running: 362 tok/s at concurrency 8, ~1085 tok/s extrapolated at the probe's
# ThreadPoolExecutor(24). 525 rows x (1200+2200) max tokens = 1.79M tokens
# -> ~27 min worst case, ~17 min at realistic reply lengths. Well inside budget;
# the 4 resident vLLM instances are not the constraint here.

# -----------------------------------------------------------------------------
# F6. The neutral prompt is INCOHERENT in production, and v2's promotion
# checklist misses it.
#
# The neutral head asserts "The user turn ends with an OPERATOR block. The
# OPERATOR states how the child must differ from the parent ... It is the only
# authority on that," and then disclaims any strength requirement of its own.
# Production has NO OPERATOR block: diff_problem_user_prompt.txt contains only
# parent_template / parent_problem / allowed_groups / allowed_skills (verified --
# the string "OPERATOR" appears nowhere in prompt_templates/ or prompts.py).
# Promoted as-is, stage 1 is pointed at a block that does not exist while every
# instruction to mutate has been deleted: the compliant reply becomes the parent.
#
# Also: FAMILY_SYSTEM_PROMPT_FILE is at prompts.py:452, not in config.py as the
# checklist states. It is a module constant, not a config field.
#
# FIX: do not promote this file. If it is ever promoted, an OPERATOR block must
# be added to diff_problem_user_prompt.txt first, and TARGET_CELL_BLOCK (appended
# AFTER the user prompt when target_cell is not None, which is the shipped
# config) must be moved before it so the OPERATOR really is last.

# -----------------------------------------------------------------------------
# F7. The negative control on skill retention is misstated.
#
# v2: "retention runs 7%-30% around a 12.5% chance baseline, with SE ~8pp at
# n~29, so no arm separates from chance ... Retention is noise, in both
# directions."
#   - SE at n=29 is 6.1pp (uniform null), not 8pp.
#   - The uniform 1/8 null is wrong: parent and child skill marginals are both
#     skewed, so the marginal-matched null is 0.1135.
#   - Pooled retention is 52/281 = 0.185, vs uniform p = 0.0037, vs the correct
#     marginal null p = 0.0004. It is not noise.
#   - Three arms clear uncorrected 0.05 individually: full_rewrite 8/27 p=.008,
#     parameter 8/29 p=.013, scale 8/29 p=.013 -- two of the three are arms v2
#     CUT. (None survive Bonferroni over 10 arms.)
# The CONCLUSION "no stage-1 operator moves the REQUIRED-skill distribution"
# still stands on the target_cell_injection argument, which I verified:
# config.py:166-167 records "Target SKILL compliance is 96-99%, GROUP 61-78%"
# with target_cell_injection = True. Only the retention leg is wrong.

# -----------------------------------------------------------------------------
# F8. The probe does not persist the raw stage-1 reply, so the parse change v2
# adds cannot be audited after the run. v1 rows carry only the PARSED fields.
# The v2 parse_plan also still diverges from production in the other direction:
# production's parse_family_plan takes the LAST non-placeholder match per key
# (prompts.py docstring: 13 of 24 base-policy replies echo the template first);
# the probe's regex takes the FIRST match and never skips a "<...>" placeholder.
# v2 fixes only the vocabulary half of the divergence.
# FIX: row["stage1_reply"] = r1, and call prompts.parse_family_plan directly
# instead of maintaining a second parser.

# -----------------------------------------------------------------------------
# VERIFIED AND CORRECT in the v2 proposal (do not re-litigate):
#   - all per-arm v1 figures (sim ALL/EXEC, exec counts, stage1/2_ok, archive
#     nearest, the 10 gate yields under its own definition) reproduce exactly;
#   - 48 champions / 29 distinct skeletons; 300 rows; 0 parents shared by all
#     10 arms (D3 is real);
#   - off-vocabulary SKILL 21/281 = 7.5%, feedback 11 / adaptive 10 / simplify 5
#     distinct skill strings, GROUP off-vocabulary 0/281;
#   - Fisher p: donor vs directed .181, vs scale .532, vs local_diff 1.000;
#     n=81 for .27 vs .10 and n=269 for .27 vs .17 at 80% power;
#   - declared skill distribution 21.0/20.3/17.8/14.9/7.1/4.6/3.9/2.8%;
#   - _split_family_system on the neutral file: 8 blocks, 827-char tail
#     (shipped 785), sentinel "Now create a child family" intact, rotation
#     preserves the five-line schema, CHANGE MADE renamed in all 4 places;
#   - every cited line number (prompts.py:452/468-478/571, evolution.py:773-783/
#     897, test_prompts.py:46/171/180/187, test_mutation_pairs.py:136,
#     test_evolution_guard.py:413, probe_diff_mutation.py:76/169/196,
#     probe_stage2_ablation.py:55) is exact, and STRUCTURAL MUTATION is read
#     nowhere in src/ except FAMILY_KEYS;
#   - the one-step donor fix `(d+1) % len(pool)` is safe: all 48 champions sit
#     in 48 distinct cells, so a collision can only occur at d == a.
#
# OVERSTATED BUT NOT FALSE: "arms shared only 5 of 21 distinct parents" is the
# MINIMUM pairwise overlap; the median is 9 and donor&structural share 10.
