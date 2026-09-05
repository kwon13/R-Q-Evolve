"""CPU-only checks for frozen inference inputs and prompt parity."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval_omni_math2_checkpoints.py"
spec = importlib.util.spec_from_file_location("omni_infer", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestFrozenInputs(unittest.TestCase):
    def test_incomplete_tail_is_preserved_not_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "predictions.jsonl"
            p.write_bytes(b'{"id": 1}\n{"id":')
            mod.recover_prediction_tail(p)
            self.assertEqual(mod.read_jsonl(p), [{"id": 1}])
            backups = list(p.parent.glob("*.incomplete-tail.*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b'{"id":')

    def test_complete_last_record_is_not_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "predictions.jsonl"
            p.write_text('{"id": 1}')
            mod.recover_prediction_tail(p)
            self.assertEqual(mod.read_jsonl(p), [{"id": 1}])
            self.assertTrue(p.read_bytes().endswith(b"\n"))

    def test_frozen_file_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "manifest.jsonl"
            mod.freeze(p, "original\n")
            mod.freeze(p, "original\n")
            with self.assertRaises(ValueError):
                mod.freeze(p, "different\n")
            self.assertEqual(p.read_text(), "original\n")

    def test_base_prompt_matches_existing_evaluator(self):
        class Tokenizer:
            chat_template = None
        expected = ("<|im_start|>user\n" + mod.SYSTEM_PROMPT +
                    "\nQuestion: What is 2+2?<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(mod.build_prompt(Tokenizer(), "What is 2+2?"), expected)

    def test_chat_prompt_does_not_include_reference(self):
        class Tokenizer:
            chat_template = "template"

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "rendered"
        tok = Tokenizer()
        self.assertEqual(mod.build_prompt(tok, "question"), "rendered")
        self.assertEqual(tok.messages, [{"role": "system", "content": mod.SYSTEM_PROMPT},
                                        {"role": "user", "content": "question"}])
        self.assertEqual(tok.kwargs, {"add_generation_prompt": True,
                                      "tokenize": False, "add_special_tokens": True})

    def test_actual_pinned_manifest_is_reproducible(self):
        if not mod.SOURCE.exists():
            self.skipTest("Pinned local dataset unavailable")
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            mod.prepare(p)
            mod.prepare(p)
            rows = mod.read_jsonl(p / "manifest.jsonl")
            self.assertEqual(len(rows), 4428)
            self.assertEqual(sum(r["eligible"] for r in rows), 4174)
            self.assertEqual(sum(r["gradable"] for r in rows), 4170)
            self.assertEqual({r["source_id"] for r in rows if r["eligible"] and not r["gradable"]},
                             {4168, 4416, 2846, 2913})
            counts = [sum(r["gradable"] and d in r["domains"] and r["problem_type"] == t
                          for r in rows) for d in mod.DOMAINS for t in mod.TYPES]
            self.assertEqual(len(counts), 35)
            self.assertEqual(min(counts), 2)


if __name__ == "__main__":
    unittest.main()
