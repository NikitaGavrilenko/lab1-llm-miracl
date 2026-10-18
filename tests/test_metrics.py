"""Metric tests use toy labels only, never produce laboratory result tables."""
import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analyze_results as a
from run_experiment import parse_scores


class ParserTests(unittest.TestCase):
    def test_accepts_independent_binary_scores(self):
        self.assertEqual(parse_scores('{"p_relevant":[0.9,0.9]}', 2), [.9, .9])

    def test_rejects_wrong_length_boolean_nan_and_range(self):
        for raw in ('{"p_relevant":[0.5]}', '{"p_relevant":[true,0]}',
                    '{"p_relevant":[NaN,0]}', '{"p_relevant":[1.1,0]}',
                    '{"p_relevant":["0.5",0]}', '[]', '{"p":[0,1]}'):
            self.assertIsNone(parse_scores(raw, 2), raw)

    def test_live_reader_only_ignores_incomplete_final_line(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.jsonl"
            path.write_bytes(b'{"x":1}\n{"x":')
            rows, ignored = a.read_jsonl(path, live=True)
            self.assertEqual((rows, ignored), ([{"x": 1}], 1))
            path.write_bytes(b'broken\n{"x":1}\n')
            with self.assertRaises(ValueError):
                a.read_jsonl(path, live=True)


class MetricTests(unittest.TestCase):
    def test_operational_macro_f1_matches_explicit_labels(self):
        y, pred = [0, 0, 1, 1, 1], [0, 2, 1, 0, 2]
        expected = f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)
        self.assertAlmostEqual(float(a.f1_from_counts(a.confusion_counts(y, pred))), expected)
        self.assertEqual(float(a.f1_from_counts(a.confusion_counts([0, 1], [2, 2]))), 0)

    def test_ranking_hand_calculation_and_invalid(self):
        scores = a.ranking_metrics([0, 1, 1], [1, .9, .8])
        ideal = 1 + 1 / math.log2(3)
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(scores["ndcg10_judged"], dcg / ideal)
        self.assertEqual(scores["mrr10_judged"], .5)
        self.assertEqual(scores["recall3_judged"], 1)
        self.assertTrue(all(x == 0 for x in a.ranking_metrics([0, 1], valid=False).values()))

    def test_ties_preserve_supplied_document_order(self):
        self.assertEqual(a.ranking_metrics([0, 1], [.5, .5])["mrr10_judged"], .5)

    def test_random_expectation_matches_all_permutations(self):
        labels = [1, 1, 0, 0]
        observed = [a.ranking_metrics(labels, order=order)
                    for order in itertools.permutations(range(4))]
        expected = a.random_expected_ranking(labels)
        for metric in expected:
            self.assertAlmostEqual(expected[metric], np.mean([r[metric] for r in observed]))

    def test_calibration_includes_endpoints(self):
        ece, bins = a.calibration([0, 1], [0., 1.])
        self.assertEqual(ece, 0)
        self.assertEqual(bins[0]["n"], 1)
        self.assertEqual(bins[-1]["n"], 1)
        self.assertEqual(sum(r["n"] for r in bins), 2)

    def test_cluster_bootstrap_deterministic_and_pointmass(self):
        counts = [a.confusion_counts([0, 1], [0, 1])] * 3
        result = a.cluster_bootstrap(counts, [[1, .5]] * 3, replicates=100)
        self.assertEqual(result["macro_f1_operation_ci95"], [1., 1.])
        self.assertEqual(result["recall3_judged_ci95"], [.5, .5])
        self.assertEqual(result, a.cluster_bootstrap(counts, [[1, .5]] * 3, replicates=100))

    def test_bm25_uses_positive_idf(self):
        docs = [{"title": "", "text": "кот кот"}, {"title": "", "text": "собака"}]
        scores = a.bm25_scores("кот", docs)
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(scores[1], 0)

    def test_resident_vram_is_specific_to_model(self):
        runtime = {"models": {"small": {"loaded": {"models": [
            {"name": "other", "size_vram": 999}, {"name": "small", "size_vram": 2 ** 30}]}}}}
        self.assertEqual(a.resident_vram(runtime, "small"), 1)
        self.assertIsNone(a.resident_vram(runtime, "absent"))


class SelectionTests(unittest.TestCase):
    def test_refuses_incomplete_then_preserves_existing_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data/prepared").mkdir(parents=True)
            (root / "results").mkdir()
            (root / "data/prepared/validation.jsonl").write_text(
                json.dumps({"qid": "q1", "docs": []}) + "\n", encoding="utf-8")
            cfg = dict(models=["A"], modes=["zero-shot", "few-shot"],
                       robust_query_count=0, splits={"validation": 1}, utility="test_formula")
            with self.assertRaises(RuntimeError):
                a.select_winner(root, cfg, [], [], {})
            selection = dict(selected_model="A", selected_mode="few-shot", threshold=.5)
            path = root / "results/selection.json"
            a.save_json(path, selection)
            before = path.read_bytes()
            out, status = a.select_winner(root, cfg, [], [], {})
            self.assertEqual(status, "existing_selection_preserved")
            self.assertEqual(out, selection)
            self.assertEqual(path.read_bytes(), before)



    def test_selection_uses_validation_resource_tradeoff_and_rejects_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data/prepared").mkdir(parents=True)
            (root / "results").mkdir()
            (root / "data/prepared/validation.jsonl").write_text(
                json.dumps({"qid": "q1", "docs": []}) + "\n", encoding="utf-8")
            cfg = dict(models=["A", "B"], modes=["zero-shot", "few-shot"],
                       robust_query_count=0, splits={"validation": 1}, utility="frozen_test_formula")
            rows, summaries = [], []
            for model in cfg["models"]:
                for mode in cfg["modes"]:
                    rows.append(dict(model=model, mode=mode, split="validation", qid="q1",
                                     variant="main", cache_key=model + mode))
                    summaries.append(dict(model=model, mode=mode, split="validation", query_count=1,
                        latency_p95_s=1 if model == "A" else 10, resident_vram_gib=1 if model == "A" else 8,
                        invalid_query_rate=0, query_coverage=1, mcc_valid=.6, brier_valid=.1,
                        macro_f1_operation=.7 if model == "A" else .9))
            with self.assertRaises(AssertionError):
                a.select_winner(root, cfg, summaries, rows + [dict(split="test")], {})
            winner, status = a.select_winner(root, cfg, summaries, rows, {})
            self.assertEqual(status, "selection_created")
            self.assertEqual(winner["selected_model"], "A")
            self.assertEqual(winner["normalization_validation_only"]["max_p95_s"], 10)
            self.assertEqual(winner["threshold"], .5)



class PriorBaselineTests(unittest.TestCase):
    def test_prior_uses_train_only_and_threshold_point_five(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data/prepared"
            data.mkdir(parents=True)
            train_docs = [dict(id=str(i), title="", text="example", rel=y)
                          for i, y in enumerate([1, 0, 0, 0])]
            (data / "train.jsonl").write_text(json.dumps(dict(qid="train1", docs=train_docs)) + "\n",
                                            encoding="utf-8")
            validation_docs = [dict(id=str(i), title="", text="example", rel=y)
                               for i, y in enumerate([1, 0])]
            (data / "validation.jsonl").write_text(json.dumps(
                dict(qid="val1", query="example", docs=validation_docs)) + "\n", encoding="utf-8")
            rows = a.baseline_results(root, ["validation"])
            prior = next(r for r in rows if r["model"] == "constant_train_prior")
            majority = next(r for r in rows if r["model"] == "majority_0")
            self.assertEqual(prior["constant_probability"], .25)
            self.assertEqual(prior["train_positive_count"], 1)
            self.assertEqual(prior["train_document_count"], 4)
            self.assertEqual(prior["threshold"], .5)
            self.assertEqual(prior["macro_f1_operation"], majority["macro_f1_operation"])
            self.assertAlmostEqual(prior["brier_valid"], .3125)
            self.assertAlmostEqual(prior["ece10_valid"], .25)
            self.assertEqual(prior["ranking_policy"], "constant_scores_stable_seed42_document_order")



if __name__ == "__main__":
    unittest.main()
