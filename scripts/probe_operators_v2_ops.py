"""Stage-1 operator set v2: 5 arms, neutral system prompt, paired design.

Drop-in replacement for the OPS/TAIL/stage1 block of scripts/probe_operators.py.
Changes from v1, each tied to a defect:

  D1  stage1() loads diff_problem_system_prompt_NEUTRAL.txt. The shipped file's
      "Create a structurally different child family" / "Changing only parameters
      ... does not count" head is gone, so the OPERATOR is the only thing in the
      context that sets mutation strength. Evidence it was needed: `parameter`,
      whose correct answer is ~1.00 by construction, measured 0.648 in v1.
  D2  rotate_shots is ON (matching config.rotate_few_shots=True) but the draw is
      keyed on the PARENT, so every arm sees the same 3 EXAMPLE blocks for the
      same parent. Example strength becomes a blocked nuisance factor instead of
      a confound, without needing a strength-balanced example pool.
  D3  parents are drawn ONCE and every arm runs the same parent list in the same
      order with the same sampling seed. Arms are paired; between-arm deltas no
      longer carry parent-draw variance.
  D4  n is 75, not 30. v1's headline was over parseable children; the honest
      denominator is EXECUTABLE children, which was 9-15 of 30. At that n no
      pair of arms separates (best contrast, donor 8/30 vs directed 3/30, is
      Fisher p=0.18). n=75 x 5 arms is the same 375-call budget as 30 x 10 and
      reaches the ~81/arm needed for a 0.27-vs-0.10 contrast at 80% power.
  --  parse: the caller must apply the PRODUCTION vocabulary check. v1 accepted
      any string as SKILL; 21 of 281 (7.5%) were off-vocabulary ("pigeonhole",
      "inclusion-exclusion", "dynamic_programming"), all of which production's
      _first_token() turns into mutation_failed.

Not fixed here (D5): stage 2 still receives the parent SOURCE in every arm.
That is a separate probe, and it is where the skill distribution actually lives.
"""

# The schema field is renamed STRUCTURAL MUTATION -> CHANGE MADE in the neutral
# prompt. The probe's own parse_plan never read that field, so nothing to change
# there; see the note in the task write-up for production's blast radius.

OPS = {
# ---- instrument check. Correct answer known a priori: skeleton sim ~1.00. ----
"parameter": """
Keep the parent's statement structure exactly. Change ONLY the numeric ranges
and constants its parameters are drawn from, so that instances differ in size
but the mathematics and the solution route are unchanged.
""",
# ---- minimal named edit: the one arm whose claim is checkable against the diff.
"local_diff": """
Change EXACTLY ONE clause of the parent family and nothing else. Name the clause
you are replacing and give its replacement. The rest of the statement -- objects,
bounds, the quantity asked for -- must survive word for word.
""",
# ---- maximal single-parent change. Under the neutral prompt this is a real arm.
"structural": """
Keep the parent's mathematical domain and its kind of object. Change the
DECISIVE SOLUTION STRUCTURE: the move that the shortest clean solution turns on
must become a different move. Changing parameters, bounds, wording, the
direction of a predicate, or the size of the object does NOT count -- a solver
who knows the parent's method must be unable to finish your child with it.
""",
# ---- the only arm that injects material from outside the parent's lineage. ----
"donor": """
A DONOR family from a different archive cell is shown after the parent. The
child must remain a mutation OF THE PARENT -- the parent's domain and objects
stay -- but it must import exactly ONE idea from the donor: a constraint, a
decisive move, or a way of counting. Name which idea you took. Copying the
donor, or ignoring it, both fail.
""",
# ---- the only closed-loop arm: mutation steered by the archive's measurements.
"feedback": """
This parent was measured. Solver success rate {s_hat:.2f} over {n_roll} rollouts;
learnability R_Q {rq:.3f}; the archive already holds {n_cell} programs in its
cell. Diagnosis: {diagnosis}

Mutate the parent to fix what the diagnosis names. Do not change anything the
diagnosis does not implicate.
""",
}

# CUT from v1, with the reason each cut is safe:
#   directed      dominated by `feedback` (weaker signal: s_hat only) and worst
#                 on every axis -- exec-restricted sim 0.942, archive-nearest
#                 0.942, gate-passing yield 3/30, the lowest measured.
#   scale         0.952 exec-restricted; definitionally a no-op for the AST
#                 skeleton. Duplicates `parameter`'s low-strength role while
#                 being a worse control (its correct answer is not known).
#   simplify      0.948; targets WELL-POSEDNESS, which this probe does not
#                 measure. Belongs in a probe scored by the blind labeller.
#   adaptive      three operators (small/medium/radical) mixed inside one arm,
#                 so n=30 was really n=10 per condition and the composition
#                 varied by draw. Its strength ladder is served identifiably by
#                 parameter < local_diff < structural as separate arms.
#   full_rewrite  the D4 casualty: 0.766 -> 0.933 once the child must run. Its
#                 novelty lived in children the policy could not implement.
#                 `donor` supplies new material the policy CAN implement
#                 (stage2_ok 28/30, the highest of the ten).
#                 TRIPWIRE: if the neutral prompt lifts full_rewrite's exec rate
#                 above ~60%, reinstate it -- the cut is about implementability,
#                 not about the idea.

TAIL = """
Hold GROUP fixed at the parent's own value: {group}. Choose SKILL freely from
the allowed list, as a DESCRIPTION of what your child actually demands -- never
as a target to write toward. Use one of the allowed SKILL words exactly as
spelled in the list.
"""


# REQUIRED ordering change in stage1(): the neutral system prompt asserts "The
# user turn ends with an OPERATOR block", so the OPERATOR must actually be last.
# v1 put TAIL after it. Build the user turn as:
#
#     user = (body
#             + "\nAllowed SKILLS:\n" + D("skill_definitions.txt")
#             + TAIL.format(group=group)
#             + "\nOPERATOR\n" + instr)
#
# and load the system prompt from diff_problem_system_prompt_neutral.txt with
# the parent-keyed 3-of-8 EXAMPLE rotation (prompts._split_family_system works
# on the neutral file unchanged: 8 blocks, tail preserved -- verified).
