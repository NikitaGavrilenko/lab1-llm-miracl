"""Small CPU fusion baselines; all fitted preprocessing uses training rows only.

Inputs are dense float32 arrays keyed by T (text), N (numeric), and I (image).
An absent key or an all-zero raw feature row denotes a missing modality. This
convention cannot distinguish a genuine all-zero observation from missing data.
Validation selects hyperparameters and checkpoints; it never updates weights.
"""

from __future__ import annotations

import copy
import time
from itertools import combinations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn


MODALITIES = ("T", "N", "I")


def _subset(X, indices):
    return {key: np.asarray(value)[indices] for key, value in X.items()}


def _nrows(X):
    if not X:
        raise ValueError("At least one modality array is required to infer batch size.")
    sizes = {len(value) for value in X.values()}
    if len(sizes) != 1:
        raise ValueError("All modality arrays must have the same row count.")
    return sizes.pop()


def _macro_f1(y, probabilities):
    return float(f1_score(y, probabilities >= 0.5, average="macro",
                          labels=[0, 1], zero_division=0))


class _Preprocessor:
    def __init__(self, X_train, modalities):
        self.modalities = tuple(modalities)
        self.scalers = {
            key: StandardScaler().fit(np.asarray(X_train[key], dtype=np.float32))
            for key in self.modalities
        }
        self.dimensions = {key: scaler.n_features_in_
                           for key, scaler in self.scalers.items()}

    def transform(self, X):
        n = _nrows(X)
        values, masks = [], []
        for key in self.modalities:
            if key in X:
                raw = np.asarray(X[key], dtype=np.float32)
                if raw.shape != (n, self.dimensions[key]):
                    raise ValueError(f"Unexpected shape for modality {key}: {raw.shape}")
                if not np.isfinite(raw).all():
                    raise ValueError(f"Non-finite features in modality {key}.")
                available = np.any(raw != 0, axis=1)
                transformed = self.scalers[key].transform(raw).astype(np.float32)
                transformed[~available] = 0
            else:
                available = np.zeros(n, dtype=bool)
                transformed = np.zeros((n, self.dimensions[key]), dtype=np.float32)
            values.append(transformed)
            masks.append(available)
        return values, np.stack(masks, axis=1)


def _set_metadata(model, start_wall, start_cpu, count, **extra):
    model.metadata = {
        "training_seconds": time.perf_counter() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
        "parameter_count": int(count),
        **extra,
    }
    model.metadata_ = model.metadata


class LogisticModel:
    def __init__(self, preprocessor, estimator):
        self.preprocessor = preprocessor
        self.estimator = estimator

    def predict_proba(self, X):
        values, _ = self.preprocessor.transform(X)
        return self.estimator.predict_proba(np.concatenate(values, axis=1))[:, 1]


def _fit_logistic(name, X, y, train_idx, val_idx, modalities, seed, tuning):
    wall, cpu = time.perf_counter(), time.process_time()
    preprocessor = _Preprocessor(_subset(X, train_idx), modalities)
    train_values, _ = preprocessor.transform(_subset(X, train_idx))
    val_values, _ = preprocessor.transform(_subset(X, val_idx))
    train = np.concatenate(train_values, axis=1)
    val = np.concatenate(val_values, axis=1)
    best, best_score, best_c = None, -np.inf, None
    for c in (0.01, 0.1, 1.0):
        estimator = LogisticRegression(C=c, max_iter=1000, solver="lbfgs",
                                       random_state=seed)
        estimator.fit(train, y[train_idx])
        score = _macro_f1(y[val_idx], estimator.predict_proba(val)[:, 1])
        tuning.append({"model": name, "C": c, "val_macro_f1": score})
        if score > best_score:
            best, best_score, best_c = estimator, score, c
    model = LogisticModel(preprocessor, best)
    _set_metadata(model, wall, cpu, best.coef_.size + best.intercept_.size,
                  selected_C=best_c, val_macro_f1=best_score,
                  modalities=list(modalities))
    return model


class LateFusionModel:
    """Average probabilities, renormalizing over available modalities per row."""

    def __init__(self, unimodal, weights, prior):
        self.unimodal = unimodal
        self.weights = np.asarray(weights, dtype=np.float64)
        self.prior = float(prior)

    def predict_proba(self, X):
        n = _nrows(X)
        probabilities = np.full((n, 3), self.prior, dtype=np.float64)
        available = np.zeros((n, 3), dtype=bool)
        for j, key in enumerate(MODALITIES):
            if key in X:
                available[:, j] = np.any(np.asarray(X[key]) != 0, axis=1)
                probabilities[:, j] = self.unimodal[key].predict_proba(X)
        effective = available * self.weights[None, :]
        # If all selected weights belong to missing inputs, use available inputs.
        zero_weight = effective.sum(axis=1) == 0
        effective[zero_weight] = available[zero_weight]
        denominator = effective.sum(axis=1)
        result = np.full(n, self.prior, dtype=np.float64)
        np.divide((probabilities * effective).sum(axis=1), denominator,
                  out=result, where=denominator > 0)
        return result


class _FusionNetwork(nn.Module):
    def __init__(self, dimensions, attention):
        super().__init__()
        self.attention = attention
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dimension, 16), nn.ReLU())
            for dimension in dimensions
        ])
        if attention:
            self.query = nn.Parameter(torch.empty(16))
            nn.init.normal_(self.query, std=0.1)
            self.head = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        else:
            self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, values, available):
        tokens = torch.stack([projection(value) for projection, value
                              in zip(self.projections, values)], dim=1)
        mask = available.clone()
        if self.training:
            keep = torch.rand(mask.shape, device=mask.device) >= 0.15
            reduced = mask & keep
            # Preserve one observed modality when random dropout removed all.
            empty = ~reduced.any(dim=1) & mask.any(dim=1)
            rows = torch.where(empty)[0]
            reduced[rows, mask[rows].long().argmax(dim=1)] = True
            mask = reduced
        tokens = tokens * mask.unsqueeze(-1)
        if self.attention:
            scores = (tokens * self.query).sum(dim=-1) / 4.0
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1) * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            representation = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        else:
            representation = tokens.flatten(start_dim=1)
        return self.head(representation).squeeze(-1)


class NeuralModel:
    def __init__(self, preprocessor, network):
        self.preprocessor = preprocessor
        self.network = network.eval()

    def predict_proba(self, X):
        values, available = self.preprocessor.transform(X)
        result = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(available), 512):
                batch = [torch.from_numpy(value[start:start + 512]) for value in values]
                mask = torch.from_numpy(available[start:start + 512])
                result.append(torch.sigmoid(self.network(batch, mask)).numpy())
        return np.concatenate(result) if result else np.empty(0, dtype=np.float32)


def _fit_neural(name, X, y, train_idx, val_idx, attention, seed, tuning):
    wall, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    preprocessor = _Preprocessor(_subset(X, train_idx), MODALITIES)
    train_values, train_mask = preprocessor.transform(_subset(X, train_idx))
    val_values, val_mask = preprocessor.transform(_subset(X, val_idx))
    train_values = [torch.from_numpy(value) for value in train_values]
    val_values = [torch.from_numpy(value) for value in val_values]
    train_mask, val_mask = torch.from_numpy(train_mask), torch.from_numpy(val_mask)
    train_y = torch.as_tensor(y[train_idx], dtype=torch.float32)
    val_y = torch.as_tensor(y[val_idx], dtype=torch.float32)
    network = _FusionNetwork([preprocessor.dimensions[key] for key in MODALITIES],
                             attention=attention)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, stale, best_epoch = np.inf, 0, 0
    best_state = copy.deepcopy(network.state_dict())
    epochs_run = 0
    for epoch in range(1, 101):
        network.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), 128):
            indices = torch.from_numpy(order[start:start + 128])
            optimizer.zero_grad(set_to_none=True)
            logits = network([value[indices] for value in train_values], train_mask[indices])
            loss = criterion(logits, train_y[indices])
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            logits = network(val_values, val_mask)
            val_loss = float(criterion(logits, val_y))
        epochs_run = epoch
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(network.state_dict())
        else:
            stale += 1
        if stale >= 10:
            break
    network.load_state_dict(best_state)
    model = NeuralModel(preprocessor, network)
    val_f1 = _macro_f1(y[val_idx], model.predict_proba(_subset(X, val_idx)))
    tuning.append({"model": name, "best_epoch": best_epoch, "epochs_run": epochs_run,
                   "val_bce": best_loss, "val_macro_f1": val_f1})
    _set_metadata(model, wall, cpu, sum(p.numel() for p in network.parameters()),
                  best_epoch=best_epoch, epochs_run=epochs_run, val_bce=best_loss,
                  val_macro_f1=val_f1, modality_dropout=0.15,
                  projection_size=16, hidden_size=32, modalities=list(MODALITIES))
    return model


def fit_models(X, y, train_idx, val_idx, seed=42):
    """Fit eight models on train; select C, late weights and epochs on validation.

    Metadata timings include hyperparameter search/checkpoint selection. Late
    fusion timings cover combination selection only: their base models are shared.
    All decisions use a fixed 0.5 probability threshold. No train+val refit occurs.
    """
    y = np.asarray(y)
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Train and validation sets must both be nonempty.")
    if np.intersect1d(train_idx, val_idx).size:
        raise ValueError("Train and validation indices must be disjoint.")
    if set(np.unique(y[train_idx])) != {0, 1}:
        raise ValueError("Training labels must contain both binary classes 0 and 1.")
    if set(X) != set(MODALITIES):
        raise ValueError("Training features must contain exactly T, N, I.")
    if _nrows(X) != len(y):
        raise ValueError("Feature and label row counts differ.")
    models, tuning = {}, []
    names = {"T": "text_only", "N": "numeric_only", "I": "image_only"}
    for key, name in names.items():
        models[name] = _fit_logistic(name, X, y, train_idx, val_idx, [key], seed, tuning)
    models["early_fusion"] = _fit_logistic(
        "early_fusion", X, y, train_idx, val_idx, MODALITIES, seed, tuning)
    unimodal = {key: models[name] for key, name in names.items()}
    base_parameters = sum(model.metadata["parameter_count"] for model in unimodal.values())
    prior = float(y[train_idx].mean())
    wall, cpu = time.perf_counter(), time.process_time()
    models["late_mean"] = LateFusionModel(unimodal, [1 / 3] * 3, prior)
    _set_metadata(models["late_mean"], wall, cpu, base_parameters, weights=[1 / 3] * 3,
                  fusion_parameter_count=0,
                  shared_base_models=list(names.values()))
    wall, cpu = time.perf_counter(), time.process_time()
    best_score, best_weights = -np.inf, None
    val_X = _subset(X, val_idx)
    # Cache base predictions: the simplex search does not retrain any estimator.
    val_probs = np.column_stack([unimodal[key].predict_proba(val_X) for key in MODALITIES])
    val_available = np.column_stack([np.any(val_X[key] != 0, axis=1)
                                     for key in MODALITIES])
    for a in range(11):
        for b in range(11 - a):
            weights = np.asarray([a, b, 10 - a - b], dtype=np.float64) / 10
            effective = val_available * weights[None, :]
            zero_weight = effective.sum(axis=1) == 0
            effective[zero_weight] = val_available[zero_weight]
            denominator = effective.sum(axis=1)
            probabilities = np.full(len(val_idx), prior, dtype=np.float64)
            np.divide((val_probs * effective).sum(axis=1), denominator,
                      out=probabilities, where=denominator > 0)
            score = _macro_f1(y[val_idx], probabilities)
            tuning.append({"model": "late_weighted", "weights": weights.tolist(),
                           "val_macro_f1": score})
            if score > best_score:
                best_score, best_weights = score, weights
    models["late_weighted"] = LateFusionModel(unimodal, best_weights, prior)
    _set_metadata(models["late_weighted"], wall, cpu, base_parameters + 2,
                  weights=best_weights.tolist(), fusion_parameter_count=2,
                  val_macro_f1=best_score, shared_base_models=list(names.values()))
    for name, attention in (("intermediate_mlp", False), ("attention_fusion", True)):
        models[name] = _fit_neural(name, X, y, train_idx, val_idx,
                                   attention, seed, tuning)
    return models, tuning


def fit_pairwise(X, y, train_idx, val_idx, seed=42):
    """Retrain TN, TI and NI early-fusion logistic models with the same C search."""
    models, tuning = {}, []
    y = np.asarray(y)
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    if np.intersect1d(train_idx, val_idx).size:
        raise ValueError("Train and validation indices must be disjoint.")
    for modalities in combinations(MODALITIES, 2):
        name = "".join(modalities)
        models[name] = _fit_logistic(name, X, y, train_idx, val_idx,
                                     modalities, seed, tuning)
    return models
