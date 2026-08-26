# Replacement OPS entries. Drop-in for the dict in scripts/probe_operators.py.
# Only the arms that do not implement the operator they are named after are
# listed; full_rewrite, structural and scale keep their current text (their
# defects are in the harness, not the wording).

OPS_REPLACEMENT = {

# ShinkaEvolve diff mode is a SEARCH/REPLACE patch applied to the artifact.
# "change exactly one clause and nothing else" is a promise; a patch is a
# mechanism. The harness applies it, so nothing outside it can drift.
"local_diff": """
Emit a PATCH against the parent family. Do not restate the family.
Use exactly this form, once:

<<<<<<< SEARCH
[the clause you are replacing, copied character for character from PARENT PROBLEM FAMILY]
=======
[its replacement]
>>>>>>> REPLACE

The SEARCH text must occur verbatim in the parent family. Everything outside it
is preserved by the harness, so do not write it out. Then give GROUP and SKILL
for the patched family.
""",

# EoH M1 is "motivated by the parent's idea", not "hit a success band". 34 of
# the 48 archive parents sit at s_hat > 0.85, so the band form collapses to one
# sentence for 71% of draws and asks for an unreachable one-hop move.
"directed": """
Name the DECISIVE REASONING MOVE the parent's shortest clean solution turns on.
Keep the parent's domain and the kind of object it is about. Motivated by that
move but not bound to it, write a child whose shortest clean solution needs the
parent's move PLUS one further step that a direct route cannot skip. Say which
move you kept and which step is new. Never mention difficulty, solver success,
or a target band in the CHILD FAMILY itself.
""",

# EoH M2 / EvoOptiGraph parameter level acts on the generator's sampling ranges.
# Those live in rng.randint(...), not in the prose family, so no stage-1 prose
# instruction can express it: this arm was a no-op at stage 1 and pure stage-2
# resampling noise thereafter (it scored 0.669, second-MOST novel of ten).
# It has to be a harness operator with no model call -- and it is then the
# design's true negative control, expected at similarity 1.000.
"parameter": None,   # see parameter_mutate() below; remove the LLM arm

# Promptbreeder hypermutation mutates the mutation prompt using fitness. Drawing
# the strength uniformly at random makes one arm a 1/3-1/3-1/3 MIXTURE of three
# operators at n~10 each, and its "radical" branch ("only the domain stays")
# contradicts TAIL's group hold -- which is why this arm held GROUP only 48% of
# the time against 93-100% elsewhere. Split it into three arms.
"adaptive_small": """
Change exactly one clause of the parent family. The solution route stays the
parent's.
""",
"adaptive_medium": """
Keep the parent's objects. Change the QUESTION asked about them, so the answer
is a different quantity.
""",
"adaptive_radical": """
Change both the objects the question is about and the decisive move of its
shortest clean solution. The mathematical domain stays {group} -- that is the
one thing you may not change.
""",

# EVOTOOL blame attribution names the COMPONENT that failed. The shipped version
# emits a 3-way if/else on s_hat, plus n_cell (the constant 1 for all 48 cells)
# and R_Q (0.000 for 30 of 48 parents): two of its three "measurements" carry no
# information, and the third takes one of three values, the same one 71% of the
# time. Attribute to a clause, computed from the rollouts the run already stores.
"feedback": """
This parent was solved by {n_solved} of {n_roll} rollouts. Of the {n_failed}
that failed, {n_wrong} returned a wrong integer and {n_timeout} ran out of
budget. Every failing rollout stopped at the same place in the statement:

    {blamed_clause}

Rewrite ONLY that clause. Quote it, give its replacement, and leave every other
clause of the parent family character for character unchanged. If the clause is
already minimal, say so and stop.
""",

# Promptbreeder EDA conditions on the POPULATION's estimated distribution, not
# on one donor. One donor is constrained crossover -- the thing the crossover
# probe already failed at (40-47% of children copied one parent wholesale).
"donor": """
Below are {k} families sampled from across the archive. They are a picture of
what this population already contains, NOT material to copy.

{population}

Write a child of the PARENT. It keeps the parent's domain and its objects, and
its decisive reasoning move must be one that none of the {k} families above
uses. Name the move you chose and name the family it is most unlike.
Reproducing any of the {k}, or ignoring them, both fail.
""",

# EoH M3 must be allowed to decline. Forcing a deletion when nothing is
# redundant turns one arm into two operators and hides the null result.
"simplify": """
Find every condition, object, or bound in the parent that the answer does NOT
depend on, and delete it. The child asks the same underlying question with the
redundant material removed. If nothing is redundant, reply
CHILD FAMILY: NONE
and stop. A forced deletion is not a simplification.
""",

# There is no control arm. The shipped order lives only in the SYSTEM prompt,
# which every arm gets, so no arm is measured against "what ships today".
"control": "",
}


def parameter_mutate(parent_source: str, rng) -> str:
    """The real EoH-M2 / parameter-level operator: no model call at all.

    Rewrites the integer bounds of the parent's sampling calls and leaves the
    problem family byte-identical. Expected structural similarity 1.000 -- it is
    the negative control the probe currently lacks.
    """
    import ast
    _SAMPLERS = {"randint", "randrange"}

    class _Widen(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in _SAMPLERS:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                        arg.value = max(1, int(arg.value * rng.uniform(1.2, 2.0)))
            return node

    tree = _Widen().visit(ast.parse(parent_source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
