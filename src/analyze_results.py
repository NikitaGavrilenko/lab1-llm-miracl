"""Analyze conditional MIRACL relevance scores; freeze selection using validation only."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPS = 2000
SEED = 42
THRESHOLD = 0.5


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path, live=False, snapshot=None):
    content = Path(path).read_bytes() if snapshot is None else snapshot
    lines = content.splitlines()
    rows, ignored = [], 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if live and index == len(lines) - 1 and not content.endswith(b"\n"):
                ignored += 1
            else:
                raise
    return rows, ignored


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def hash_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def scores_valid(row):
    p = row.get("probabilities")
    return bool(row.get("valid")) and isinstance(p, list) and len(p) == len(row["labels"]) and all(
        not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
        for v in p)


def predictions(row):
    return np.asarray([int(p >= THRESHOLD) for p in row["probabilities"]] if scores_valid(row)
                      else [2] * len(row["labels"]), dtype=int)


def confusion_counts(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return np.asarray([value for label in (0, 1) for value in (
        np.sum((y == label) & (pred == label)),
        np.sum((y != label) & (pred == label)),
        np.sum((y == label) & (pred != label)))], dtype=float)


def f1_from_counts(counts):
    counts = np.asarray(counts)
    numerators = 2 * counts[..., (0, 3)]
    denominators = numerators + counts[..., (1, 4)] + counts[..., (2, 5)]
    values = np.divide(numerators, denominators, out=np.zeros_like(numerators, dtype=float),
                       where=denominators != 0)
    return values.mean(axis=-1)


def ranking_metrics(labels, scores=None, valid=True, order=None):
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    assert positives > 0, "Query without positive qrels"
    if not valid:
        return dict(ndcg10_judged=0.0, mrr10_judged=0.0, recall10_judged=0.0, recall3_judged=0.0)
    if order is None:
        order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ranked = labels[np.asarray(order, dtype=int)]
    discounts = 1 / np.log2(np.arange(2, min(10, len(labels)) + 2))
    ideal = discounts[:min(positives, 10)].sum()
    positions = np.flatnonzero(ranked[:10])
    return dict(ndcg10_judged=float((ranked[:10] * discounts).sum() / ideal),
                mrr10_judged=float(1 / (positions[0] + 1)) if len(positions) else 0.0,
                recall10_judged=float(ranked[:10].sum() / positives),
                recall3_judged=float(ranked[:3].sum() / positives))


def random_expected_ranking(labels):
    n, m = len(labels), int(sum(labels))
    discounts = 1 / np.log2(np.arange(2, min(10, n) + 2))
    expected_rr = sum(math.comb(n - rank, m - 1) / math.comb(n, m) / rank
                      for rank in range(1, min(10, n) + 1) if n - rank >= m - 1)
    return dict(ndcg10_judged=float((m / n) * discounts.sum() / discounts[:min(m, 10)].sum()),
                mrr10_judged=float(expected_rr), recall10_judged=min(10, n) / n,
                recall3_judged=min(3, n) / n)


def calibration(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if not len(y):
        return None, []
    ids = np.minimum((p * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        mask = ids == b
        count = int(mask.sum())
        mean_p = float(p[mask].mean()) if count else None
        positive_rate = float(y[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(mean_p - positive_rate)
        rows.append(dict(bin=b, lower=b / bins, upper=(b + 1) / bins, n=count,
                         mean_probability=mean_p, observed_positive_fraction=positive_rate))
    return float(ece), rows


def cluster_bootstrap(counts, ranking, replicates=BOOTSTRAPS):
    counts = np.asarray(counts, dtype=float)
    ranking = np.asarray(ranking, dtype=float)
    n = len(counts)
    if not n:
        return {}
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, n, size=(replicates, n))
    f1 = f1_from_counts(counts[indices].sum(axis=1))
    values = {"macro_f1_operation": f1}
    for index, name in enumerate(("ndcg10_judged", "recall3_judged")):
        values[name] = ranking[indices, index].mean(axis=1)
    return {name + "_ci95": [float(v) for v in np.quantile(scores, [.025, .975])]
            for name, scores in values.items()}


def query_values(row):
    y, pred = row["labels"], predictions(row)
    ranking = ranking_metrics(y, row.get("probabilities"), scores_valid(row))
    return confusion_counts(y, pred), ranking


def summarize(rows, runtime, with_ci=True):
    counts, ranks, ys, preds, valid_y, valid_p = [], [], [], [], [], []
    latency, eval_rates, prompt_tokens, token_counts = [], [], [], []
    for row in rows:
        count, rank = query_values(row)
        counts.append(count)
        ranks.append(rank)
        ys.extend(row["labels"])
        preds.extend(predictions(row).tolist())
        if scores_valid(row):
            valid_y.extend(row["labels"])
            valid_p.extend(row["probabilities"])
        if isinstance(row.get("latency_s"), (int, float)):
            latency.append(row["latency_s"])
        response = row.get("response", {})
        if response.get("eval_duration", 0) > 0:
            eval_rates.append(response.get("eval_count", 0) / (response["eval_duration"] / 1e9))
        if response.get("prompt_eval_count") is not None:
            prompt_tokens.append(response["prompt_eval_count"])
        token_counts.append(response.get("eval_count", 0))
    vy, vp = np.asarray(valid_y), np.asarray(valid_p, dtype=float)
    valid_pred = (vp >= THRESHOLD).astype(int)
    ece, bins = calibration(vy, vp)
    model = rows[0]["model"]
    out = dict(model=model, mode=rows[0]["mode"], split=rows[0]["split"], variant="main",
               query_count=len(rows), document_count=len(ys),
               valid_query_count=sum(scores_valid(r) for r in rows),
               valid_document_count=len(vy),
               query_coverage=sum(scores_valid(r) for r in rows) / len(rows),
               document_coverage=len(vy) / len(ys),
               invalid_query_rate=1 - sum(scores_valid(r) for r in rows) / len(rows),
               accuracy_operation=float(np.mean(np.asarray(ys) == np.asarray(preds))),
               macro_f1_operation=float(f1_from_counts(np.sum(counts, axis=0))),
               macro_f1_valid=float(f1_from_counts(confusion_counts(vy, valid_pred))) if len(vy) else None,
               mcc_valid=float(matthews_corrcoef(vy, valid_pred)) if len(vy) else None,
               brier_valid=float(np.mean((vp - vy) ** 2)) if len(vy) else None,
               ece10_valid=ece,
               auroc_valid=float(roc_auc_score(vy, vp)) if len(set(valid_y)) == 2 else None,
               latency_p50_s=float(np.median(latency)) if latency else None,
               latency_p95_s=float(np.quantile(latency, .95)) if latency else None,
               latency_observed_count=len(latency),
               generation_tokens_per_s_median=float(np.median(eval_rates)) if eval_rates else None,
               prompt_tokens_max=max(prompt_tokens) if prompt_tokens else None,
               generated_tokens_total=sum(token_counts),
               resident_vram_gib=resident_vram(runtime, model),
               confusion=dict(zip(("tp0", "fp0", "fn0", "tp1", "fp1", "fn1"),
                                  map(int, np.sum(counts, axis=0)))))
    tp0, fp0, fn0, tp1, fp1, fn1 = np.sum(counts, axis=0)
    ratio = lambda num, den: float(num / den) if den else 0.0
    out.update(precision_relevant_operation=ratio(tp1, tp1 + fp1),
               recall_relevant_operation=ratio(tp1, tp1 + fn1),
               f1_relevant_operation=ratio(2 * tp1, 2 * tp1 + fp1 + fn1),
               specificity_operation=ratio(tp0, tp0 + fn0),
               balanced_accuracy_operation=(ratio(tp0, tp0 + fn0) + ratio(tp1, tp1 + fn1)) / 2,
               confusion_true01_pred012=[[int(tp0), int(fp1), int(fn0 - fp1)],
                                         [int(fp0), int(tp1), int(fn1 - fp0)]])
    for key in ranks[0]:
        out[key] = float(np.mean([r[key] for r in ranks]))
    if with_ci:
        out.update(cluster_bootstrap(counts, [[r["ndcg10_judged"], r["recall3_judged"]] for r in ranks]))
    return out, bins


def resident_vram(runtime, model):
    info = runtime.get("models", {}).get(model, {})
    candidates = info.get("loaded", {}).get("models", [])
    matches = [r for r in candidates if r.get("name") == model or r.get("model") == model]
    if len(matches) != 1 or not isinstance(matches[0].get("size_vram"), (int, float)):
        return None
    return matches[0]["size_vram"] / (1024 ** 3)


def bm25_scores(query, docs):
    tokenize = lambda text: re.findall(r"\w+", text.casefold())
    tokens = [tokenize(d["title"][:120] + " " + d["text"]) for d in docs]
    n, avg_len = len(tokens), np.mean([len(t) for t in tokens])
    frequencies = [Counter(t) for t in tokens]
    result = np.zeros(n, dtype=float)
    if not avg_len:
        return result
    for term in set(tokenize(query)):
        df = sum(term in freq for freq in frequencies)
        idf = math.log(1 + (n - df + .5) / (df + .5))
        for index, freq in enumerate(frequencies):
            tf = freq[term]
            result[index] += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * len(tokens[index]) / avg_len))
    return result


def prepared_rows(root, split):
    return read_jsonl(root / "data/prepared" / (split + ".jsonl"))[0]


def ordered_docs(row):
    import random
    docs = sorted(row["docs"], key=lambda d: d["id"])
    random.Random("42:" + row["qid"]).shuffle(docs)
    return docs


def baseline_results(root, splits):
    train = prepared_rows(root, "train")
    train_labels = [d["rel"] for row in train for d in row["docs"]]
    assert train_labels and all(label in (0, 1) for label in train_labels)
    train_prior = sum(train_labels) / len(train_labels)
    output = []
    for split in splits:
        rows = prepared_rows(root, split)
        majority_records, prior_records, bm25_ranks, random_ranks = [], [], [], []
        for row in rows:
            docs = ordered_docs(row)
            limit = min(900, 14000 // len(docs))
            visible = [{**d, "text": d["text"][:limit]} for d in docs]
            labels = [d["rel"] for d in docs]
            majority_records.append(dict(model="majority_0", mode="baseline", split=split,
                                         labels=labels, probabilities=[0.0] * len(docs), valid=True))
            prior_records.append(dict(model="constant_train_prior", mode="baseline", split=split,
                                      labels=labels, probabilities=[train_prior] * len(docs), valid=True))
            bm25_ranks.append(ranking_metrics(labels, bm25_scores(row["query"], visible)))
            random_ranks.append(random_expected_ranking(labels))
        majority, _ = summarize(majority_records, {}, with_ci=True)
        # A majority classifier has no ranking preference; report stable tie order explicitly.
        majority["ranking_policy"] = "all_zero_scores_stable_seed42_document_order"
        output.append(majority)
        prior, _ = summarize(prior_records, {}, with_ci=True)
        prior.update(constant_probability=train_prior, probability_source="pooled_train_labels_only",
                     train_positive_count=sum(train_labels), train_document_count=len(train_labels),
                     threshold=THRESHOLD, ranking_policy="constant_scores_stable_seed42_document_order")
        output.append(prior)
        for name, ranks in (("BM25_same_judged_candidates", bm25_ranks), ("random_ranking_exact_expectation", random_ranks)):
            result = dict(model=name, mode="baseline", split=split, query_count=len(rows),
                          document_count=sum(len(r["docs"]) for r in rows), variant="main")
            result.update({k: float(np.mean([r[k] for r in ranks])) for k in ranks[0]})
            result["scope"] = "conditional judged pool, same visible title/text; no full-corpus claim"
            output.append(result)
    return output


def validate_records(root, records):
    prepared = {split: {r["qid"]: r for r in prepared_rows(root, split)}
                for split in sorted({r["split"] for r in records})}
    unique, duplicate_keys = {}, []
    for row in records:
        assert row["split"] in prepared and row["qid"] in prepared[row["split"]], "Unknown split/qid"
        source = prepared[row["split"]][row["qid"]]
        byid = {d["id"]: d for d in source["docs"]}
        assert len(row["docids"]) == len(set(row["docids"])) == len(byid), "Candidate count mismatch"
        assert set(row["docids"]) == set(byid), "Candidate ids mismatch"
        assert row["labels"] == [byid[d]["rel"] for d in row["docids"]], "Gold labels mismatch"
        assert all(label in (0, 1) for label in row["labels"]), "Nonbinary labels"
        assert bool(row["valid"]) == scores_valid(row), "Invalid recorded probabilities"
        key = (row["model"], row["mode"], row["split"], row["qid"], row["variant"])
        if key in unique:
            assert unique[key]["cache_key"] == row["cache_key"], "Conflicting protocols for same example"
            assert unique[key] == row, "Conflicting duplicate cache record"
            duplicate_keys.append(key)
        unique[key] = row
    return list(unique.values()), len(duplicate_keys)


def paired_comparisons(groups):
    output = []
    for model, split in sorted({(m, s) for m, mode, s in groups}):
        zero = {r["qid"]: r for r in groups.get((model, "zero-shot", split), [])}
        few = {r["qid"]: r for r in groups.get((model, "few-shot", split), [])}
        qids = sorted(set(zero) & set(few))
        if not qids:
            continue
        zc, fc, zr, fr = [], [], [], []
        for qid in qids:
            a, ar = query_values(zero[qid])
            b, br = query_values(few[qid])
            zc.append(a); fc.append(b)
            zr.append([ar["ndcg10_judged"], ar["recall3_judged"]])
            fr.append([br["ndcg10_judged"], br["recall3_judged"]])
        zc, fc, zr, fr = map(np.asarray, (zc, fc, zr, fr))
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(qids), (BOOTSTRAPS, len(qids)))
        deltas = {"macro_f1_operation": f1_from_counts(fc[draws].sum(axis=1)) - f1_from_counts(zc[draws].sum(axis=1)),
                  "ndcg10_judged": (fr[draws, 0] - zr[draws, 0]).mean(axis=1),
                  "recall3_judged": (fr[draws, 1] - zr[draws, 1]).mean(axis=1)}
        estimate = {"macro_f1_operation": float(f1_from_counts(fc.sum(axis=0)) - f1_from_counts(zc.sum(axis=0))),
                    "ndcg10_judged": float((fr[:, 0] - zr[:, 0]).mean()),
                    "recall3_judged": float((fr[:, 1] - zr[:, 1]).mean())}
        for metric, distribution in deltas.items():
            lo, hi = np.quantile(distribution, [.025, .975])
            output.append(dict(model=model, split=split, contrast="few-shot_minus_zero-shot",
                               metric=metric, shared_queries=len(qids), delta=estimate[metric],
                               ci95_low=float(lo), ci95_high=float(hi),
                               exploratory=True, multiplicity_adjustment="none"))
    return output


def query_type(query):
    text = query.casefold().strip()
    if re.search(r"\b(кто|кем|кого|чей|чья|чьи)\b", text):
        return "who"
    if re.search(r"\b(когда|году|год|дата|веке)\b", text):
        return "when"
    if re.search(r"\b(где|куда|откуда)\b", text):
        return "where"
    return "other"


def errors_and_slices(root, groups):
    output, errors = [], []
    for (model, mode, split), rows in sorted(groups.items()):
        data = {r["qid"]: r for r in prepared_rows(root, split)}
        slices = defaultdict(list)
        for row in rows:
            source = data[row["qid"]]
            byid = {d["id"]: d for d in source["docs"]}
            pred = predictions(row)
            for index, docid in enumerate(row["docids"]):
                doc, y = byid[docid], row["labels"][index]
                p = row["probabilities"][index] if scores_valid(row) else None
                length = len(doc["text"])
                length_bin = "<=500" if length <= 500 else "501-900" if length <= 900 else ">900"
                item = (y, int(pred[index]), p)
                for category, label in (("true_class", str(y)), ("original_text_chars", length_bin),
                                        ("query_type_heuristic", query_type(source["query"]))):
                    slices[(category, label)].append(item)
                if p is not None and pred[index] != y:
                    errors.append(dict(model=model, mode=mode, split=split, qid=row["qid"], docid=docid,
                        query=source["query"], title=doc["title"], text=doc["text"][:row["passage_limit_chars"]],
                        original_text_chars=length, true_label=y, predicted_label=int(pred[index]),
                        p_relevant=p, wrong_class_confidence=p if pred[index] == 1 else 1 - p,
                        input_conditioning="score conditioned jointly on all query candidates"))
        for (category, label), items in slices.items():
            y, pred = np.asarray([i[0] for i in items]), np.asarray([i[1] for i in items])
            valid = pred != 2
            output.append(dict(model=model, mode=mode, split=split, slice_type=category, slice_value=label,
                documents=len(items), coverage=float(valid.mean()), error_rate_operation=float((pred != y).mean()),
                macro_f1_operation=float(f1_from_counts(confusion_counts(y, pred))),
                brier_valid=float(np.mean([(p - y0) ** 2 for y0, _, p in items if p is not None]))
                if valid.any() else None))
    errors.sort(key=lambda x: (-x["wrong_class_confidence"], x["model"], x["qid"], x["docid"]))
    # Keep balanced error examples by model/mode/split rather than one large-model list.
    selected, seen = [], Counter()
    for row in errors:
        key = (row["model"], row["mode"], row["split"])
        if seen[key] < 5:
            selected.append(row); seen[key] += 1
    return output, selected


def robustness(records):
    by_key = {(r["model"], r["split"], r["mode"], r["qid"], r["variant"]): r for r in records}
    pairs = defaultdict(list)
    for row in records:
        if row["variant"] != "paraphrase":
            continue
        key = (row["model"], row["split"], row["mode"], row["qid"], "main")
        if key in by_key:
            pairs[(row["model"], row["split"], row["mode"])].append((by_key[key], row))
    output = []
    for (model, split, mode), rows in sorted(pairs.items()):
        good = [(a, b) for a, b in rows if scores_valid(a) and scores_valid(b)]
        flips, deltas = 0, []
        for a, b in good:
            assert a["docids"] == b["docids"], "Paraphrase candidate order changed"
            flips += int(np.sum(predictions(a) != predictions(b)))
            deltas.extend(np.abs(np.asarray(a["probabilities"]) - b["probabilities"]).tolist())
        main, _ = summarize([a for a, b in rows], {}, with_ci=False)
        paraphrase, _ = summarize([b for a, b in rows], {}, with_ci=False)
        output.append(dict(model=model, split=split, mode=mode, paired_queries=len(rows),
            both_valid_queries=len(good), paired_valid_documents=len(deltas),
            prediction_flips=flips, flip_rate_valid=flips / len(deltas) if deltas else None,
            mean_absolute_probability_delta=float(np.mean(deltas)) if deltas else None,
            main_macro_f1_operation=main["macro_f1_operation"],
            paraphrase_macro_f1_operation=paraphrase["macro_f1_operation"],
            macro_f1_delta=paraphrase["macro_f1_operation"] - main["macro_f1_operation"],
            main_ndcg10_judged=main["ndcg10_judged"], paraphrase_ndcg10_judged=paraphrase["ndcg10_judged"],
            note="Prompt paraphrase, not query rewrite; fixed first 8 queries; metrics operational on all pairs"))
    return output


def completeness(root, cfg, records, split="validation"):
    qids = [r["qid"] for r in prepared_rows(root, split)]
    observed = {(r["model"], r["mode"], r["qid"], r["variant"]) for r in records if r["split"] == split}
    models, modes = cfg["models"], cfg["modes"]
    selection_path = root / "results/selection.json"
    if split == "application" and selection_path.exists():
        selected = read_json(selection_path)
        models, modes = [selected["selected_model"]], [selected["selected_mode"]]
    expected = {(m, mode, q, "main") for m in models for mode in modes for q in qids}
    if "zero-shot" in modes:
        expected |= {(m, "zero-shot", q, "paraphrase") for m in models
                     for q in qids[:cfg["robust_query_count"]]}
    missing = sorted(expected - observed)
    return dict(split=split, expected=len(expected), observed=len(observed & expected),
                complete=not missing, missing_count=len(missing), missing=missing)


def select_winner(root, cfg, summaries, records, runtime):
    path = root / "results/selection.json"
    if path.exists():
        return read_json(path), "existing_selection_preserved"
    assert not any(r["split"] in ("test", "application") for r in records), \
        "Selection must be frozen before test/application inference"
    check = completeness(root, cfg, records)
    if not check["complete"]:
        raise RuntimeError("Validation incomplete: " + str(check["observed"]) + "/" + str(check["expected"]))
    rows = [dict(s) for s in summaries if s["split"] == "validation" and s["model"] in cfg["models"]]
    assert len(rows) == len(cfg["models"]) * len(cfg["modes"])
    assert all(s["query_count"] == cfg["splits"]["validation"] for s in rows)
    assert all(s["latency_p95_s"] is not None and s["resident_vram_gib"] is not None for s in rows), \
        "Missing latency or model-specific resident VRAM"
    max_p95 = max(s["latency_p95_s"] for s in rows)
    max_vram = max(s["resident_vram_gib"] for s in rows)
    assert max_p95 > 0 and max_vram > 0
    for row in rows:
        row["eligible"] = row["invalid_query_rate"] <= .05
        row["utility"] = None if row["mcc_valid"] is None or row["brier_valid"] is None else (
            .55 * row["macro_f1_operation"] + .15 * (row["mcc_valid"] + 1) / 2 +
            .15 * (1 - row["brier_valid"]) + .10 * (1 - row["latency_p95_s"] / max_p95) +
            .05 * (1 - row["resident_vram_gib"] / max_vram))
    eligible = [r for r in rows if r["eligible"] and r["utility"] is not None]
    rule = "invalid_query_rate <= 0.05, then maximum preregistered utility"
    if not eligible:
        coverage = max(r["query_coverage"] for r in rows)
        eligible = [r for r in rows if r["query_coverage"] == coverage]
        rule = "No eligible candidate: maximum query coverage, then utility if defined"
    eligible.sort(key=lambda r: (-(r["utility"] if r["utility"] is not None else -1),
                                 r["latency_p95_s"], r["model"], r["mode"]))
    best = eligible[0]
    validation_records = [r for r in records if r["split"] == "validation"]
    selection = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected_model=best["model"],
        selected_mode=best["mode"], threshold=THRESHOLD, selection_split="validation", rule=rule,
        utility=best["utility"], utility_formula=cfg["utility"],
        normalization_validation_only=dict(max_p95_s=max_p95, max_resident_vram_gib=max_vram),
        protocol_sha256=hash_json(cfg),
        validation_prediction_keys_sha256=hash_json(sorted(r["cache_key"] for r in validation_records)),
        validation_completeness=check, candidates=rows, selected_metrics=best,
        caveat="Scores are jointly conditioned on judged candidates; VRAM is resident api/ps, not peak.")
    save_json(path, selection)
    return selection, "selection_created"


def figures(root, groups, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    target = root / "figures"
    target.mkdir(exist_ok=True)
    for split in sorted({s["split"] for s in groups}):
        rows = [s for s in groups if s["split"] == split]
        models = list(dict.fromkeys(s["model"] for s in rows))
        y = np.arange(len(models))
        fig, axes = plt.subplots(1, 2, figsize=(12, max(3, len(models) * .54)), sharey=True)
        for ax, metric, title in zip(axes, ("macro_f1_operation", "ndcg10_judged"),
                                     ("Классификация: Macro-F1", "Reranking: nDCG@10")):
            for mode, offset, color in (("zero-shot", -.17, "#3B73AF"), ("few-shot", .17, "#DA8545")):
                values = [next((s for s in rows if s["model"] == m and s["mode"] == mode), None) for m in models]
                for index, s in enumerate(values):
                    if s is None:
                        continue
                    ci = s.get(metric + "_ci95", [s[metric], s[metric]])
                    ax.barh(index + offset, s[metric], .30, color=color,
                            label=mode if index == 0 else None)
                    ax.plot(ci, [index + offset] * 2, color="#202020", linewidth=1)
            ax.set_xlim(0, 1); ax.set_title(title); ax.grid(axis="x", alpha=.2)
        axes[0].set_yticks(y, models); axes[0].invert_yaxis()
        axes[1].legend(loc="lower right")
        fig.suptitle(split + ": 95% query-cluster CI; judged pool")
        fig.tight_layout(); fig.savefig(target / ("comparison_" + split + ".png"), dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, max(3, len(models) * .54)))
        for mode, offset, color in (("zero-shot", -.17, "#3B73AF"), ("few-shot", .17, "#DA8545")):
            for index, model in enumerate(models):
                s = next((s for s in rows if s["model"] == model and s["mode"] == mode), None)
                if s and s["latency_p95_s"] is not None:
                    ax.barh(index + offset, s["latency_p95_s"], .3, color=color, alpha=.45,
                            label=mode + " p95" if index == 0 else None)
                    ax.barh(index + offset, s["latency_p50_s"], .3, color=color,
                            label=mode + " p50" if index == 0 else None)
        ax.set_yticks(y, models); ax.invert_yaxis(); ax.set_xlabel("Секунды на один запрос со всеми кандидатами")
        ax.set_title(split + ": latency, warm inference"); ax.legend(); ax.grid(axis="x", alpha=.2)
        fig.tight_layout(); fig.savefig(target / ("latency_" + split + ".png"), dpi=160); plt.close(fig)


def calibration_figure(root, bin_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for split in sorted({r["split"] for r in bin_rows}):
        models = list(dict.fromkeys(r["model"] for r in bin_rows if r["split"] == split))
        columns, nrows = 2, math.ceil(len(models) / 2)
        fig, axes = plt.subplots(nrows, columns, figsize=(10, max(3.2, 2.7 * nrows)), squeeze=False)
        for ax, model in zip(axes.flat, models):
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            for mode, color in (("zero-shot", "#3B73AF"), ("few-shot", "#DA8545")):
                rows = [r for r in bin_rows if r["split"] == split and r["model"] == model
                        and r["mode"] == mode and r["n"] > 0]
                if rows:
                    ax.plot([r["mean_probability"] for r in rows],
                            [r["observed_positive_fraction"] for r in rows], "o-", color=color, label=mode)
            ax.set_title(model); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2); ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(models):]:
            ax.axis("off")
        fig.supxlabel("Self-verbalized p(relevant)"); fig.supylabel("Observed positive fraction")
        fig.suptitle(split + ": calibration on valid responses, 10 equal-width bins")
        fig.tight_layout(); fig.savefig(root / "figures" / ("calibration_" + split + ".png"), dpi=160)
        plt.close(fig)


def run(args):
    root = args.root
    cfg = read_json(root / "configs/protocol.json")
    runtime = read_json(root / "results/runtime.json")
    raw_snapshot = (root / "results/predictions.jsonl").read_bytes()
    records, ignored = read_jsonl(root / "results/predictions.jsonl", live=True, snapshot=raw_snapshot)
    excluded_models = Counter(r["model"] for r in records if r["model"] not in cfg["models"])
    records = [r for r in records if r["model"] in cfg["models"]]
    records, duplicates = validate_records(root, records)
    assert records, "No predictions"
    grouped = defaultdict(list)
    for row in records:
        if row["variant"] == "main":
            grouped[(row["model"], row["mode"], row["split"])].append(row)
    summaries, bin_rows = [], []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r["qid"])
        summary, bins = summarize(rows, runtime)
        summaries.append(summary)
        bin_rows.extend([{**dict(model=key[0], mode=key[1], split=key[2]), **r} for r in bins])
    outputs = root / "results"
    checks = dict(prediction_records=len(records), ignored_live_final_line=ignored,
        identical_duplicate_records=duplicates, excluded_model_records=dict(excluded_models),
        completeness=[completeness(root, cfg, records, split) for split in sorted({r["split"] for r in records})],
        definitions=dict(threshold=THRESHOLD, bootstrap_replicates=BOOTSTRAPS, bootstrap_unit="qid",
            scores="self-verbalized, jointly conditioned on all candidates; not token posteriors",
            operational="Abstention=2; Macro-F1 labels [0,1], zero_division=0; invalid incurs false negatives",
            calibration="Brier/ECE/AUROC/MCC valid responses only; coverage reported",
            ranking="binary qrels on fixed judged pool; invalid empty ranking scores zero",
            latency="wall time per query, all candidate scores in one call; failures without response have missing latency",
            resource="model-specific resident size_vram from api/ps; not peak GPU memory",
            uncertainty="Bootstrap captures query sampling uncertainty conditional on fixed model, prompt and corpus",
            baselines="Full prepared split only; compare to model groups only when counts are complete"),
        raw_file_sha256_snapshot=hashlib.sha256(raw_snapshot).hexdigest())
    baseline = baseline_results(root, sorted({r["split"] for r in records}))
    summary_output = dict(protocol_version=cfg["protocol_version"], definitions=checks["definitions"],
                          groups=summaries, baselines=baseline)
    save_json(outputs / "summary.json", summary_output); save_csv(outputs / "summary.csv", summaries + baseline)
    paired = paired_comparisons(grouped)
    slices, errors = errors_and_slices(root, grouped)
    robust = robustness(records)
    for name, value in (("paired_comparisons", paired), ("error_slices", slices), ("error_examples", errors),
                         ("robustness", robust), ("calibration_bins", bin_rows)):
        save_json(outputs / (name + ".json"), value); save_csv(outputs / (name + ".csv"), value)
    if args.select:
        selection, status = select_winner(root, cfg, summaries, records, runtime)
        checks["selection_status"] = status
        print(json.dumps({"selection_status": status, "model": selection["selected_model"],
                          "mode": selection["selected_mode"]}, ensure_ascii=False))
    elif (outputs / "selection.json").exists():
        checks["selection_status"] = "existing_selection_preserved"
    else:
        checks["selection_status"] = "not_requested"
    save_json(outputs / "analysis_checks.json", checks)
    if not args.no_figures:
        figures(root, summaries, baseline); calibration_figure(root, bin_rows)
    print(json.dumps({"groups": len(summaries), "records": len(records),
                      "validation_complete": completeness(root, cfg, records)["complete"],
                      "selection": checks["selection_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--select", action="store_true", help="Freeze validation winner only after all 8 models finish")
    parser.add_argument("--no-figures", action="store_true")
    run(parser.parse_args())
