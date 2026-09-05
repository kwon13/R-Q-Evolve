"""Offline judgment tests: no SDK, credentials, GPU, or network required."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "judge_omni_math2", Path(__file__).resolve().parents[1] / "scripts/judge_omni_math2.py")
j = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(j)


def fixture(ident=0, answer="2", response="The answer is 2.", **overrides):
    item = {"id": ident, "problem": "What is 1+1?", "answer": answer,
            "problem_sha256": j.sha256("What is 1+1?"), "eligible": True,
            "gradable": True, "qa_flags": [], "domains": ["Algebra"], "problem_type": "function"}
    item.update(overrides)
    return item, {"id": ident, "response": response, "response_sha256": j.sha256(response)}


def reply(correct=True, **overrides):
    obj = {"correct": correct, "extracted_final_answer": "2", "reasoning": "Equivalent."}
    obj.update(overrides)
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(obj)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                      "completion_tokens_details": {"reasoning_tokens": 12}}}


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, payload):
        self.calls.append(payload)
        return reply()

    def judge(self, call=None, **kwargs):
        return j.Judge(self.root / "cache", call or self.client, sleep=lambda _: None, **kwargs)

    def test_all_response_semantics_and_strict_payload(self):
        out = self.judge().judge(*fixture())
        self.assertEqual(out["status"], "ok")
        self.assertIs(out["correct"], True)
        payload = self.calls[0]
        self.assertEqual(payload["model"], "gpt-5-mini-2025-08-07")
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertEqual(payload["max_completion_tokens"], 8192)
        self.assertNotIn("temperature", payload)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(set(json.loads(payload["messages"][1]["content"])),
                         {"question", "reference_answer", "candidate_response"})

    def test_cache_deduplicates_ids_and_rejects_changed_reference(self):
        judge = self.judge()
        first = judge.judge(*fixture())
        second = judge.judge(*fixture(ident=9))
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["id"], 9)
        self.assertEqual(len(self.calls), 1)
        judge.judge(*fixture(answer="3"))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(first["cache_key"], second["cache_key"])

    def test_cache_invalidation_when_settings_change(self):
        self.judge().judge(*fixture())
        self.judge(max_completion_tokens=16384).judge(*fixture())
        self.assertEqual(len(self.calls), 2)

    def test_concurrent_duplicate_uses_single_request(self):
        def slow(payload):
            self.calls.append(payload)
            time.sleep(0.03)
            return reply()
        judge = self.judge(slow)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(lambda i: judge.judge(*fixture(ident=i)), [0, 1]))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(sum(o["cache_hit"] for o in outputs), 1)

    def test_ungradable_and_empty_reference_never_sent(self):
        for changes in ({"gradable": False}, {"eligible": False}, {"answer": ""}):
            out = self.judge().judge(*fixture(**changes))
            self.assertEqual(out["status"], "ungradable")
            self.assertIsNone(out["correct"])
        self.assertEqual(self.calls, [])

    def test_wrong_hash_stops_before_api(self):
        item, prediction = fixture()
        prediction["response_sha256"] = "stale"
        with self.assertRaises(ValueError):
            self.judge().judge(item, prediction)
        self.assertEqual(self.calls, [])

    def test_string_boolean_is_error_and_not_cached(self):
        count = []
        def bad(payload):
            count.append(payload)
            return reply(correct="yes")
        out = self.judge(bad, max_attempts=2).judge(*fixture())
        self.assertEqual(out["status"], "error")
        self.assertIsNone(out["correct"])
        self.assertEqual(out["error"], "invalid_judgment_schema")
        self.assertEqual(len(count), 2)
        self.assertEqual(list((self.root / "cache").glob("*.json")), [])

    def test_length_retries_with_larger_bounded_budget(self):
        def length_then_ok(payload):
            self.calls.append(payload)
            if len(self.calls) == 1:
                return {"choices": [{"finish_reason": "length", "message": {"content": ""}}],
                        "usage": {"completion_tokens": 8192}}
            return reply()
        out = self.judge(length_then_ok).judge(*fixture())
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["attempts"], 2)
        self.assertEqual([c["max_completion_tokens"] for c in self.calls], [8192, 16384])
        self.assertEqual(len(out["attempt_usage"]), 2)

    def test_refusal_is_not_a_model_error(self):
        def refusal(payload):
            return {"choices": [{"finish_reason": "stop", "message": {
                "refusal": "Refused", "content": None}}]}
        out = self.judge(refusal).judge(*fixture())
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "judge_refusal")
        self.assertIsNone(out["correct"])
        self.assertEqual(out["attempts"], 1)

    def test_unknown_exception_text_never_persisted(self):
        def unsafe(payload):
            raise RuntimeError("SECRET_NOT_FOR_LOGS")
        out = self.judge(unsafe, max_attempts=1).judge(*fixture())
        self.assertEqual(out["error"], "unexpected_client_error")
        self.assertNotIn("SECRET_NOT_FOR_LOGS", j.canonical(out))

    def test_fatal_failure_stops_submissions(self):
        def no_quota(payload):
            self.calls.append(payload)
            raise j.JudgeFailure("insufficient_quota", fatal=True)
        results, fatal = j.process_pending([fixture(i) for i in range(5)], self.judge(no_quota),
                                           self.root / "out.jsonl", workers=1)
        self.assertTrue(fatal)
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["correct"])

    def test_resume_requires_hash_and_success(self):
        out = self.judge().judge(*fixture())
        self.assertTrue(j.resumable(out, out["input_sha256"]))
        self.assertFalse(j.resumable(out, "changed"))
        self.assertFalse(j.resumable(dict(out, status="error"), out["input_sha256"]))

    def test_partial_prediction_tail_only_allowed_when_following(self):
        path = self.root / "partial.jsonl"
        path.write_bytes(b'{"id":0}\n{"id":')
        self.assertEqual(j.read_jsonl(path, allow_partial_tail=True), [{"id": 0}])
        with self.assertRaises(ValueError):
            j.read_jsonl(path)

    def test_interrupted_journal_tail_preserved_and_recovered(self):
        path = self.root / "judgements.jsonl"
        path.write_bytes(b'{"id":0}\n{"id":')
        j.recover_journal_tail(path)
        self.assertEqual(j.read_jsonl(path), [{"id": 0}])
        backups = list(self.root.glob("judgements.jsonl.incomplete.*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b'{"id":')

    def test_newline_terminated_corruption_not_silently_removed(self):
        path = self.root / "judgements.jsonl"
        path.write_bytes(b'{"id":0}\n{"id":\n')
        j.recover_journal_tail(path)
        with self.assertRaises(ValueError):
            j.read_jsonl(path)

    def test_cli_resume_uses_no_additional_api_calls(self):
        item, prediction = fixture()
        j.append_record(self.root / "manifest.jsonl", item)
        j.append_record(self.root / "runs/test/predictions.jsonl", prediction)
        args = argparse.Namespace(root=self.root, run="test", timeout=1,
            max_completion_tokens=8192, max_output_ceiling=32768, max_attempts=4,
            follow=False, limit=None, workers=1, poll_seconds=1)
        self.assertEqual(j.run(args, call=self.client), 0)
        self.assertEqual(j.run(args, call=self.client), 0)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(j.read_jsonl(self.root / "runs/test/judgements.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
