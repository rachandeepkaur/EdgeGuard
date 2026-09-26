"""
TimingCNN (comparison only, added 2026-09-25): a small causal 1D CNN over
per-window timing bins, requested to compare against v1's percentile-rank
timing checks and defender/ml_baseline.py's logistic regression.

NOT part of the shipped detector -- same footprint reasoning as
ml_baseline.py (this needs PyTorch; v1/v2 deliberately don't).

Scope, stated honestly: this is a SINGLE-WINDOW causal CNN (it looks inside
one window's own 100ms bins), not the multi-window/4-second causal history
sketched in the team's neural-architecture proposal. That cross-window
version is a real, separate next step -- out of scope for today's smoke
test and real-data run.

Feature representation, per window (see bin_timing_features()):
    Split the window into N_BINS equal time bins (100ms each, for a 1.0s
    window). For each bin, three channels:
      0. total frame count (any ID)          -- overall traffic-rate shape
      1. unique CAN ID count                 -- fuzzing shows up here
      2. frame count on the watched IDs      -- where real attacks target
    Counts are log1p-transformed for scale stability before the network
    sees them. This is deliberately close to Stage 1's own signals (frame
    counts/IDs), so it isolates "does within-window temporal SHAPE help,
    not just aggregate counts" -- Stage 1 only ever sees the aggregate.
"""

import json
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from defender.threshold import choose_threshold, make_decision
from shared.schemas import DefenderOutput, TrafficWindow

N_BINS = 10
N_CHANNELS = 3


def bin_timing_features(window: TrafficWindow, watch_ids=("0D0", "6E0"),
                        n_bins: int = N_BINS) -> List[List[float]]:
    """(N_CHANNELS, n_bins) nested list of log1p-scaled counts."""
    import math as _math
    duration = window.window_end - window.window_start
    if duration <= 0:
        raise ValueError(f"window {window.window_id} has non-positive duration")
    bin_s = duration / n_bins
    watch = {w.upper() for w in watch_ids}

    total = [0] * n_bins
    ids_seen: List[set] = [set() for _ in range(n_bins)]
    watched = [0] * n_bins
    for frame in window.frames:
        offset = frame.timestamp - window.window_start
        idx = min(int(offset / bin_s), n_bins - 1) if bin_s > 0 else 0
        idx = max(0, min(idx, n_bins - 1))
        total[idx] += 1
        can_id = frame.can_id.upper()
        ids_seen[idx].add(can_id)
        if can_id in watch:
            watched[idx] += 1

    unique_counts = [len(s) for s in ids_seen]
    return [[_math.log1p(v) for v in total],
            [_math.log1p(v) for v in unique_counts],
            [_math.log1p(v) for v in watched]]


class TimingCNNNet(nn.Module):
    """Two causal-style Conv1d blocks (kernel 3, dilations 1/2) + global
    average pool + a linear head -- the small-first-pass shape from the
    team's architecture proposal, scaled down for a same-day smoke test."""

    def __init__(self, in_channels: int = N_CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=2, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)   # raw logits, shape (batch,)


class TimingCNNModel:
    """Trains/saves/loads a TimingCNNNet, plus the operating threshold --
    same choose_threshold discipline as v1/v2 (defender/threshold.py):
    calibrated on NORMAL validation scores only."""

    def __init__(self):
        self.net = TimingCNNNet()
        self.fitted = False
        self.threshold = 1.0
        self.training_windows = 0
        self.training_attack_windows = 0
        self.final_train_loss = None

    def _to_tensor(self, feature_rows: List[List[List[float]]]) -> torch.Tensor:
        return torch.tensor(feature_rows, dtype=torch.float32)

    def fit(self, feature_rows: List[List[List[float]]], labels: List[bool],
           epochs: int = 30, lr: float = 0.01, seed: int = 0) -> "TimingCNNModel":
        if len(feature_rows) != len(labels):
            raise ValueError("feature_rows and labels must be the same length")
        if len(feature_rows) < 2:
            raise ValueError("fit() needs at least 2 examples")
        if len(set(labels)) < 2:
            raise ValueError("fit() needs at least one normal and one attack example")

        torch.manual_seed(seed)
        x = self._to_tensor(feature_rows)
        y = torch.tensor([1.0 if lbl else 0.0 for lbl in labels], dtype=torch.float32)

        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.net.train()
        final_loss = None
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.net(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        self.net.eval()
        self.final_train_loss = final_loss
        self.training_windows = len(feature_rows)
        self.training_attack_windows = sum(1 for lbl in labels if lbl)
        self.fitted = True
        return self

    def predict_proba_attack(self, feature_rows: List[List[List[float]]]) -> List[float]:
        if not self.fitted:
            raise RuntimeError("TimingCNNModel is not trained. Call fit() first.")
        with torch.no_grad():
            logits = self.net(self._to_tensor(feature_rows))
            return torch.sigmoid(logits).tolist()

    def calibrate_threshold(self, normal_feature_rows: List[List[List[float]]],
                            max_false_alarm_rate: float) -> float:
        scores = self.predict_proba_attack(normal_feature_rows)
        self.threshold = choose_threshold(scores, max_false_alarm_rate)
        return self.threshold

    def save(self, path) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an untrained TimingCNNModel")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / "model.pt")
        meta = {
            "n_bins": N_BINS, "n_channels": N_CHANNELS,
            "threshold": self.threshold,
            "training_windows": self.training_windows,
            "training_attack_windows": self.training_attack_windows,
            "final_train_loss": self.final_train_loss,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "TimingCNNModel":
        directory = Path(path)
        meta_file = directory / "meta.json"
        model_file = directory / "model.pt"
        if not meta_file.exists() or not model_file.exists():
            raise FileNotFoundError(f"No TimingCNN model at {directory}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        model = cls()
        model.net.load_state_dict(torch.load(model_file, map_location="cpu"))
        model.net.eval()
        model.threshold = meta["threshold"]
        model.training_windows = meta["training_windows"]
        model.training_attack_windows = meta["training_attack_windows"]
        model.final_train_loss = meta.get("final_train_loss")
        model.fitted = True
        return model


class TimingCNNDefender:
    """Same score_window(window) -> DefenderOutput interface as
    defender.defender.Defender and defender.ml_baseline.MLBaselineDefender,
    so it drops into the exact same evaluation machinery unchanged."""

    def __init__(self, model: TimingCNNModel, watch_ids=("0D0", "6E0"),
                model_version: str = "timing_cnn_v1"):
        if not model.fitted:
            raise ValueError("TimingCNNDefender needs a fitted TimingCNNModel")
        self.model = model
        self.watch_ids = watch_ids
        self.model_version = model_version

    def score_window(self, window: TrafficWindow) -> DefenderOutput:
        start = time.perf_counter()
        features = bin_timing_features(window, self.watch_ids)
        score = self.model.predict_proba_attack([features])[0]
        latency_ms = (time.perf_counter() - start) * 1000
        decision = make_decision(score, self.model.threshold)
        evidence = "TimingCNN (causal Conv1d over 100ms timing bins)"
        return DefenderOutput(
            window_id=window.window_id, attack_score=score,
            threshold=self.model.threshold, decision=decision,
            evidence=evidence, model_version=self.model_version,
            latency_ms=latency_ms,
        )
