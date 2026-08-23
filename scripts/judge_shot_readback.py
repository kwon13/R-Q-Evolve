#!/usr/bin/env python
"""Read-back test: does a zero-shot GPT judge assign a shot child its declared cell?

The stage-1 shots are the curriculum prior -- a base policy copies them -- so a
shot whose SKILL an independent reader cannot recover from the problem text
seeds cells the archive does not really have. This measures that directly:
each CASE below instantiates one shot's CHILD FAMILY twice, and the judge sees
ONLY the rendered problem and its reference answer (never the source, never the
declared labels), mirroring the setup that produced
analysis/concept_grid_8b_8gpu/*/labels/rq_unique_programs.jsonl.

Edit CASES when the shots in prompt_templates/diff_problem_system_prompt.txt
change; answers must be recomputed by hand or script, never guessed.

Baselines: live archive read-back GROUP 80% / SKILL 17% / valid 78%.
Shots v3 (2026-08-23): GROUP 14/16, SKILL 11/16, valid 16/16.
"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_vllm_math import _load_dotenv  # noqa: E402
_load_dotenv(ROOT / ".env")
from openai import OpenAI  # noqa: E402

GROUPS = (ROOT / "prompt_templates/group_definitions.txt").read_text()
SKILLS = (ROOT / "prompt_templates/skill_definitions.txt").read_text()
G = ["number_theory", "combinatorics", "sequence", "algebra", "geometry", "inequality"]
S = ["casework", "induction", "contradiction", "invariant", "extremal_principle",
     "counting", "transformation", "construction"]

SYS = f"""You classify one competition-math problem from its statement and reference answer alone.

GROUP definitions:
{GROUPS}
SKILL definitions:
{SKILLS}
Solve the problem internally first. GROUP and SKILL must describe the shortest clean solution.
If no listed SKILL genuinely applies, answer "other" for skill.
Also judge validity: the requested answer must exist, be finite, unique, integer-valued, and fully determined by the statement.

Reply with ONLY a JSON object: {{"group": <one of {G}>, "skill": <one of {S + ['other']}>, "valid": true|false, "invalid_reason": null|"answer_in_statement"|"underdetermined"|"ill_defined"|"unsatisfiable_premise", "note": "<one sentence>"}}"""

# (declared_group, declared_skill, [ (question, answer), (question, answer) ])
CASES = [
    ("algebra", "casework", [
        ("Let b = 20 and c = 5 be positive integers. How many integers x satisfy |x^2 - 20| = 5? State only the integer.", "2"),
        ("Let b = 10 and c = 6 be positive integers. How many integers x satisfy |x^2 - 10| = 6? State only the integer.", "4")]),
    ("geometry", "induction", [
        ("Let n = 10. In the plane, 10 lines are in general position: no two are parallel and no three pass through one point. Into how many regions do these lines divide the plane? State only the integer.", "56"),
        ("Let n = 7. In the plane, 7 lines are in general position: no two are parallel and no three pass through one point. Into how many regions do these lines divide the plane? State only the integer.", "29")]),
    ("sequence", "contradiction", [
        ("A sequence of positive integers a_1, a_2, ..., a_K satisfies a_1 = 72, and each a_(k+1) is a proper divisor of a_k, meaning a_(k+1) divides a_k and a_(k+1) < a_k. What is the greatest possible value of K? State only the integer.", "6"),
        ("A sequence of positive integers a_1, a_2, ..., a_K satisfies a_1 = 240, and each a_(k+1) is a proper divisor of a_k, meaning a_(k+1) divides a_k and a_(k+1) < a_k. What is the greatest possible value of K? State only the integer.", "7")]),
    ("algebra", "invariant", [
        ("The numbers 1 through 6 are written on a blackboard. In each move, two numbers a and b are erased and the single number a + b + a*b is written. After 5 moves one number remains. Determine that number. State only the integer.", "5039"),
        ("The numbers 1 through 5 are written on a blackboard. In each move, two numbers a and b are erased and the single number a + b + a*b is written. After 4 moves one number remains. Determine that number. State only the integer.", "719")]),
    ("inequality", "construction", [
        ("Let n = 4 and s = 10 be positive integers with n < s. Positive integers a_1, ..., a_4 have sum 10. Find the greatest integer L such that in every such collection, the largest of the numbers is at least L. State only the integer.", "3"),
        ("Let n = 5 and s = 23 be positive integers with n < s. Positive integers a_1, ..., a_5 have sum 23. Find the greatest integer L such that in every such collection, the largest of the numbers is at least L. State only the integer.", "5")]),
    ("combinatorics", "counting", [
        ("There are 6 people and 6 distinct hats, with each person owning exactly one hat. In how many assignments of the hats does no person receive their own hat? State only the integer.", "265"),
        ("There are 5 people and 5 distinct hats, with each person owning exactly one hat. In how many assignments of the hats does no person receive their own hat? State only the integer.", "44")]),
    ("number_theory", "transformation", [
        ("Let N = 23 be a positive integer. How many ordered pairs of positive integers (x, y) satisfy xy + x + y = N? State only the integer.", "6"),
        ("Let N = 59 be a positive integer. How many ordered pairs of positive integers (x, y) satisfy xy + x + y = N? State only the integer.", "10")]),
    ("combinatorics", "extremal_principle", [
        ("Let n = 10 be a positive integer. Find the maximum number of edges in a simple graph with 10 vertices that contains no triangle. State only the integer.", "25"),
        ("Let n = 7 be a positive integer. Find the maximum number of edges in a simple graph with 7 vertices that contains no triangle. State only the integer.", "12")]),
]

client = OpenAI()
MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.6-luna")

def judge(item):
    dg, ds, (q, ans) = item
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": f"PROBLEM:\n{q}\n\nREFERENCE ANSWER: {ans}"}])
    txt = r.choices[0].message.content or ""
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        # LaTeX in the free-text note (\(, \le, ...) is not a JSON escape.
        d = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", txt))
    return dg, ds, q, d

jobs = [(dg, ds, inst) for dg, ds, insts in CASES for inst in insts]
with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(judge, jobs))

g_ok = s_ok = v_ok = 0
print(f"{'선언':<32}{'GPT 판독':<32}{'valid':<7}")
for dg, ds, q, d in results:
    gm, sm = d.get("group") == dg, d.get("skill") == ds
    g_ok += gm; s_ok += sm; v_ok += bool(d.get("valid"))
    mark = "✓✓" if gm and sm else ("G✓" if gm else ("S✓" if sm else "✗✗"))
    print(f"{dg+'/'+ds:<32}{str(d.get('group'))+'/'+str(d.get('skill')):<32}{str(d.get('valid')):<7}{mark}")
    if not (gm and sm and d.get("valid")):
        print(f"   └ {d.get('note','')[:120]}")
n = len(results)
print(f"\nGROUP {g_ok}/{n}   SKILL {s_ok}/{n}   valid {v_ok}/{n}")
