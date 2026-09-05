"""Offline synthetic checks. All artifacts stay inside TemporaryDirectory."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_omni_math2_comparison.py"
SPEC = importlib.util.spec_from_file_location("omni_plot_under_test", SCRIPT)
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def row(index, *, problem=None, domains=None, kind="function", eligible=True, gradable=True):
    problem = problem or f"Synthetic offline test question {index}"
    return {
        "id": index, "problem": problem, "answer": "unused",
        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "domains": ["Algebra"] if domains is None else domains,
        "problem_type": kind, "eligible": eligible, "gradable": gradable,
    }


def manifest(*rows):
    return {str(r["id"]): r for r in rows}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")


class OmniComparisonTests(unittest.TestCase):
    def test_wilson_zero_and_two_successes(self):
        self.assertEqual(plot.wilson_interval(0, 0), (None, None))
        lower, upper = plot.wilson_interval(2, 2)
        self.assertAlmostEqual(lower, 0.3423802275)
        self.assertAlmostEqual(upper, 1)
        lower, upper = plot.wilson_interval(0, 2)
        self.assertAlmostEqual(lower, 0)
        self.assertGreater(upper, .65)

    def test_no_hidden_support_filter_and_na_not_zero(self):
        rows, summary = plot.compare(manifest(row(1)), {"1": False}, {"1": True}, bootstrap=30)
        self.assertEqual(len(rows), 35)
        observed = [r for r in rows if r["n"]]
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["n"], 1)
        self.assertEqual(observed[0]["rzero_acc"], 0)
        self.assertEqual(observed[0]["rq_acc"], 1)
        self.assertTrue(observed[0]["small_n_warning"])
        self.assertTrue(all(r["rq_acc"] is None for r in rows if not r["n"]))
        self.assertFalse(summary["full_35_cell_ranking_observable"])
        self.assertEqual(summary["metrics_scope"], "observed_1_cells_only")

    def test_final_requires_unmapped_rows_too(self):
        source = manifest(row(1), row(2, kind=None))
        with self.assertRaisesRegex(ValueError, "Incomplete paired"):
            plot.compare(source, {"1": True}, {"1": True, "2": False}, bootstrap=5)
        _, summary = plot.compare(source, {"1": True, "2": True}, {"1": True, "2": False}, bootstrap=5)
        self.assertEqual(summary["paired_rows"], 2)
        self.assertEqual(summary["unmapped_rows"], 1)
        self.assertEqual(summary["overall_rq_acc"], .5)

    def test_partial_intersection_is_explicit_diagnostic(self):
        source = manifest(row(1), row(2), row(3))
        rows, summary = plot.compare(source, {"1": True, "2": False}, {"1": False, "3": True}, bootstrap=10, allow_partial=True)
        self.assertTrue(summary["diagnostic_partial"])
        self.assertEqual(summary["paired_rows"], 1)
        self.assertEqual(sum(r["n"] for r in rows), 1)

    def test_ineligible_and_ungradable_do_not_enter_denominator(self):
        source = manifest(row(1), row(2, eligible=False), row(3, gradable=False))
        _, summary = plot.compare(source, {"1": False}, {"1": True}, bootstrap=10)
        self.assertEqual(summary["eligible_gradable_rows"], 1)
        self.assertEqual(summary["overall_rq_acc"], 1)

    def test_multidomain_memberships_are_expanded_not_independent_problems(self):
        source = manifest(row(1, domains=["Algebra", "Geometry"]))
        rows, summary = plot.compare(source, {"1": True}, {"1": True}, bootstrap=10)
        self.assertEqual(summary["mapped_rows"], 1)
        self.assertEqual(summary["overlapping_cell_memberships"], 2)
        self.assertEqual(summary["unique_problem_clusters"], 1)
        self.assertEqual(summary["tied_cells"], 2)
        for r in rows:
            if r["n"]:
                self.assertEqual(r["delta_ci_low_pp"], 0)
                self.assertEqual(r["delta_ci_high_pp"], 0)

    def test_exact_duplicate_cluster_weight_sensitivity(self):
        source = manifest(row(1, problem="same"), row(2, problem="same"), row(3, problem="other"))
        rows, summary = plot.compare(source, {"1": False, "2": False, "3": True}, {"1": True, "2": True, "3": False}, bootstrap=30)
        observed = next(r for r in rows if r["n"])
        self.assertEqual(observed["n"], 3)
        self.assertEqual(observed["unique_problems"], 2)
        self.assertAlmostEqual(observed["delta_pp"], 100/3)
        self.assertEqual(observed["deduplicated_delta_pp"], 0)
        self.assertEqual(summary["unique_problem_clusters"], 2)

    def test_swapping_models_reverses_deltas_and_interval_endpoints(self):
        source = manifest(
            row(1, problem="duplicate", domains=["Algebra", "Geometry"]),
            row(2, problem="duplicate", domains=["Algebra", "Geometry"]),
            row(3, domains=["Algebra"]), row(4, domains=["Geometry"]),
            row(5, domains=["Geometry"]), row(6, domains=["Algebra"]),
        )
        left = {"1": True, "2": False, "3": False, "4": True, "5": True, "6": False}
        right = {"1": False, "2": False, "3": True, "4": False, "5": True, "6": True}
        forward, summary_f = plot.compare(source, left, right, bootstrap=500, seed=42)
        reverse, summary_r = plot.compare(source, right, left, bootstrap=500, seed=42)
        for a, b in zip(forward, reverse):
            self.assertEqual(a["n"], b["n"])
            self.assertEqual(a["unique_problems"], b["unique_problems"])
            self.assertEqual(a["delta_ci_valid_replicates"], b["delta_ci_valid_replicates"])
            if not a["n"]:
                continue
            self.assertAlmostEqual(a["delta_pp"], -b["delta_pp"])
            self.assertAlmostEqual(a["deduplicated_delta_pp"], -b["deduplicated_delta_pp"])
            self.assertAlmostEqual(a["delta_ci_low_pp"], -b["delta_ci_high_pp"])
            self.assertAlmostEqual(a["delta_ci_high_pp"], -b["delta_ci_low_pp"])
            self.assertEqual(a["rzero_wilson_low"], b["rq_wilson_low"])
            self.assertEqual(a["rzero_wilson_high"], b["rq_wilson_high"])
        for key, forward_metric in summary_f["metrics"].items():
            reverse_metric = summary_r["metrics"][key]
            self.assertAlmostEqual(forward_metric["delta_pp"], -reverse_metric["delta_pp"])
            self.assertAlmostEqual(forward_metric["conditional_delta_ci_low_pp"], -reverse_metric["conditional_delta_ci_high_pp"])
            self.assertAlmostEqual(forward_metric["conditional_delta_ci_high_pp"], -reverse_metric["conditional_delta_ci_low_pp"])

    def test_two_all_incorrect_answers_have_nonzero_absolute_uncertainty(self):
        rows, _ = plot.compare(manifest(row(1), row(2)), {"1": False, "2": False}, {"1": False, "2": False}, bootstrap=100)
        observed = next(r for r in rows if r["n"])
        self.assertEqual(observed["n"], 2)
        self.assertEqual(observed["rzero_acc"], 0)
        self.assertEqual(observed["rq_acc"], 0)
        self.assertGreater(observed["rzero_wilson_high"], .65)
        self.assertGreater(observed["rq_wilson_high"], .65)
        self.assertEqual(observed["delta_pp"], 0)

    def test_all35_with_ties_and_non_degenerate_small_n_intervals(self):
        source = {}
        for index, (domain, kind) in enumerate(plot.CELLS):
            source.update(manifest(row(index, domains=[domain], kind=kind)))
        left = {key: False for key in source}
        right = {key: True for key in source}
        rows, summary = plot.compare(source, left, right, bootstrap=30)
        self.assertTrue(summary["full_35_cell_ranking_observable"])
        self.assertEqual(summary["lower_tail_cell_count"], 7)
        self.assertEqual(len(summary["extrema"]["rq"]["observed_lowest"]), 35)
        self.assertTrue(all(r["rq_wilson_low"] < 1 for r in rows))
        # Sparse cells can disappear in joint resamples; do not invent CIs.
        self.assertEqual(summary["metrics"]["equal_cell_macro"]["conditional_ci_valid_replicates"], 0)
        self.assertIsNone(summary["metrics"]["equal_cell_macro"]["conditional_delta_ci_low_pp"])

    def test_manifest_normalization_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"manifest.jsonl"
            item = row(1, domains=["number_theory", "Number Theory"])
            write_jsonl(path, [item])
            parsed = plot.load_manifest(path)
            self.assertEqual(parsed["1"]["domains"], ["Number Theory"])
            item["problem"] = "changed"
            write_jsonl(path, [item])
            with self.assertRaisesRegex(ValueError, "Problem hash mismatch"):
                plot.load_manifest(path)

    def test_run_journal_latest_and_stale_hash(self):
        source = manifest(row(1), row(2))
        digest = hashlib.sha256(b"test response").hexdigest()
        predictions = [{"id": i, "response": "test response", "response_sha256": digest, "finish_reason": "stop"} for i in (1, 2)]
        journal = [
            {"id": 1, "response_sha256": digest, "status": "error", "correct": None},
            {"id": 1, "response_sha256": digest, "status": "ok", "correct": True},
            {"id": 2, "response_sha256": "stale", "status": "ok", "correct": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_jsonl(directory/"predictions.jsonl", predictions)
            write_jsonl(directory/"judgements.jsonl", journal)
            scores, audit = plot.load_run(directory, source)
            self.assertEqual(scores, {"1": True})
            self.assertEqual(audit["failure_counts"], {"stale_judgement_hash": 1})

    def test_corrupt_prediction_and_duplicate_manifest_fail(self):
        source = manifest(row(1))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_jsonl(directory/"predictions.jsonl", [{"id": 1, "response": "corrupt", "response_sha256": "bad"}])
            write_jsonl(directory/"judgements.jsonl", [])
            with self.assertRaisesRegex(ValueError, "Corrupt prediction"):
                plot.load_run(directory, source)
            write_jsonl(directory/"manifest.jsonl", [row(1), row(1)])
            with self.assertRaisesRegex(ValueError, "Duplicate id"):
                plot.load_manifest(directory/"manifest.jsonl")

    def test_judge_fingerprint_catches_changed_answer(self):
        item = row(1)
        prediction = {"id": 1, "response": "test response", "response_sha256": plot.sha256("test response")}
        settings = {"model": "synthetic-offline-test"}
        cache_key = plot.sha256(plot.canonical([item["problem"], item["answer"], prediction["response"], settings]))
        fingerprint = plot.sha256(plot.canonical([cache_key, True, True, None]))
        judgement = {"id": 1, "response_sha256": prediction["response_sha256"], "status": "ok", "correct": True, "judge_settings": settings, "input_sha256": fingerprint}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_jsonl(directory/"predictions.jsonl", [prediction])
            write_jsonl(directory/"judgements.jsonl", [judgement])
            scores, audit = plot.load_run(directory, manifest(item))
            self.assertEqual(scores, {"1": True})
            self.assertEqual(audit["missing_judge_provenance"], 0)
            item["answer"] = "a different answer"
            scores, audit = plot.load_run(directory, manifest(item))
            self.assertEqual(scores, {})
            self.assertEqual(audit["failure_counts"], {"stale_judgement_input": 1})

    def test_protocol_mismatch_rejected(self):
        config = {"manifest_sha256": "test-hash", "sampling": {"max_tokens": 4096}, "eos_token_id": 0, "system_prompt": "test", "chat_template_sha256": "same-template", "prompt_probe_sha256": "same", "packages": {"numpy": "test"}}
        audit = {"run": "test", "missing_judge_provenance": 0, "judge_protocols": [{"model": "synthetic"}]}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            a, b = directory/"a", directory/"b"
            a.mkdir()
            b.mkdir()
            (a/"inference_config.json").write_text(json.dumps(config))
            (b/"inference_config.json").write_text(json.dumps(config))
            verified = plot.verify_protocol_pair(a, b, "test-hash", [audit, audit])
            self.assertTrue(verified["generation_settings_equal"])
            config["sampling"]["max_tokens"] = 8192
            (b/"inference_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Inference protocol mismatch"):
                plot.verify_protocol_pair(a, b, "test-hash", [audit, audit])
            config["sampling"]["max_tokens"] = 4096
            config["chat_template_sha256"] = "changed-template"
            (b/"inference_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "chat_template_sha256"):
                plot.verify_protocol_pair(a, b, "test-hash", [audit, audit])

    def test_prediction_config_fingerprint_catches_stale_or_missing_provenance(self):
        source = manifest(row(1))
        config = {"sampling": {"max_tokens": 4096}, "model_files": {"weights": "original"}}
        prediction = {"id": 1, "response": "test", "response_sha256": plot.sha256("test")}
        judgment = {"id": 1, "response_sha256": prediction["response_sha256"], "status": "ok", "correct": True}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config_path = directory/"inference_config.json"
            config_path.write_text(json.dumps(config))
            config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            prediction["inference_config_sha256"] = config_hash
            write_jsonl(directory/"predictions.jsonl", [prediction])
            write_jsonl(directory/"judgements.jsonl", [judgment])
            scores, audit = plot.load_run(directory, source)
            self.assertEqual(scores, {"1": True})
            self.assertEqual(audit["inference_config_sha256"], config_hash)
            # Identical displayed generation settings cannot hide a changed
            # model/provenance file behind old successful predictions.
            config["model_files"]["weights"] = "changed"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Prediction inference config hash mismatch"):
                plot.load_run(directory, source)
            prediction.pop("inference_config_sha256")
            write_jsonl(directory/"predictions.jsonl", [prediction])
            with self.assertRaisesRegex(ValueError, "Prediction inference config hash mismatch"):
                plot.load_run(directory, source)

    def test_empty_paired_data_fail(self):
        with self.assertRaisesRegex(ValueError, "No successfully judged"):
            plot.compare(manifest(row(1)), {}, {}, bootstrap=5, allow_partial=True)

    def test_report_and_figures_smoke_in_temporary_directory_only(self):
        source = manifest(row(1), row(2))
        rows, summary = plot.compare(source, {"1": True, "2": False}, {"1": True, "2": True}, bootstrap=10)
        with tempfile.TemporaryDirectory(prefix="omni_plot_test_") as directory:
            output = Path(directory)
            plot.write_report(rows, summary, output)
            plot.plot_figures(rows, summary, output, "SYNTHETIC OFFLINE TEST")
            for stem in ("all35_performance_map", "all35_accuracy_intervals"):
                for extension in ("svg", "pdf", "png"):
                    self.assertTrue((output/f"{stem}.{extension}").is_file())
            loaded = json.loads((output/"summary.json").read_text())
            self.assertEqual(loaded["observed_cells"], 1)


if __name__ == "__main__":
    unittest.main()
