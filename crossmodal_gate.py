"""
crossmodal_gate.py
Finite-Sample Conformal Cross-Modal Verification Gate.
Couples continuous thermodynamic ODE boundaries, rolling-boxcar variance
normalization, and soft-bounded statistical feature residuals.
"""

from typing import Optional, Tuple
import numpy as np

import config


class CrossModalGate:
    def __init__(
        self,
        ablation: str = "full_cross_modal",
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        ridge_alpha: Optional[float] = None,
    ):
        self.ablation = ablation
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_mean = target_mean
        self.target_std = target_std

        if ridge_alpha is None:
            self.ridge_alpha = float(getattr(config, "RIDGE_ALPHA", 1.0))
        else:
            self.ridge_alpha = float(ridge_alpha)

        self.threshold: Optional[float] = None
        self.calibrated: bool = False

        # Statistical ridge regression weights
        self.weights: Optional[np.ndarray] = None

        # Identified thermodynamic parameters: dT = beta_1 * I_eff^2 + beta_2 * T + beta_0
        self.beta_1: float = 0.005
        self.beta_2: float = -0.02
        self.beta_0: float = 0.50

        # Baseline single-step residual standard deviation
        self.dyn_scale: float = 0.05
        self.stat_scale: float = 0.05

        # Rolling boxcar window length
        self.window_size: int = 15

        # Split boundaries stored as (start, end) tuples
        self.train_idx: Tuple[int, int] = (0, 0)
        self.cal_idx: Tuple[int, int] = (0, 0)
        self.val_idx: Tuple[int, int] = (0, 0)
        self.test_idx: Tuple[int, int] = (0, 0)

        self.v_aligned: np.ndarray = np.empty(0, dtype=np.float64)
        self.i_aligned: np.ndarray = np.empty(0, dtype=np.float64)
        self.t_aligned: np.ndarray = np.empty(0, dtype=np.float64)

        self.X_all: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.y_all: np.ndarray = np.empty(0, dtype=np.float64)

        self.r_th = float(getattr(config, "BASE_R_TH", 0.50))
        self.c_th = float(getattr(config, "BASE_C_TH", 100.0))
        self.r_w = float(getattr(config, "BASE_R_W", 0.50))
        self.t_ambient = float(getattr(config, "AMBIENT_TEMPERATURE", getattr(config, "T_AMBIENT", 25.0)))
        self.dt = float(getattr(config, "DT", 1.0))

        # Simulator saturation bounds
        self.t_clip_min: float = self.t_ambient - 10.0
        self.t_clip_max: float = 400.0

    @property
    def train_indices(self) -> np.ndarray:
        return np.arange(self.train_idx[0], self.train_idx[1], dtype=np.int64)

    @property
    def cal_indices(self) -> np.ndarray:
        return np.arange(self.cal_idx[0], self.cal_idx[1], dtype=np.int64)

    @property
    def val_indices(self) -> np.ndarray:
        return np.arange(self.val_idx[0], self.val_idx[1], dtype=np.int64)

    @property
    def test_indices(self) -> np.ndarray:
        return np.arange(self.test_idx[0], self.test_idx[1], dtype=np.int64)

    def _expected_dim(self) -> int:
        dims = {
            "full_cross_modal": 6,
            "missing_current": 4,
            "missing_voltage": 4,
            "no_temporal_lag": 4,
            "single_modal_ar": 2,
        }
        if self.ablation not in dims:
            raise ValueError(f"Unknown ablation mode: {self.ablation}")
        return dims[self.ablation]

    def _build_features(
        self, v: np.ndarray, i: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        v = np.asarray(v, dtype=np.float64)
        i = np.asarray(i, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)

        v_prev = v[:-1]
        i_prev = i[:-1]
        t_prev = t[:-1]
        y = t[1:]

        delta_v = np.diff(v)
        delta_i = np.diff(i)
        power_prev = v_prev * i_prev

        if self.ablation == "full_cross_modal":
            X = np.column_stack([t_prev, v_prev, i_prev, power_prev, delta_v, delta_i])
        elif self.ablation == "missing_current":
            X = np.column_stack([t_prev, v_prev, delta_v, v_prev**2])
        elif self.ablation == "missing_voltage":
            X = np.column_stack([t_prev, i_prev, delta_i, i_prev**2])
        elif self.ablation == "no_temporal_lag":
            X = np.column_stack([t_prev, v_prev, i_prev, power_prev])
        elif self.ablation == "single_modal_ar":
            X = np.column_stack([t_prev, t_prev - self.t_ambient])
        else:
            raise ValueError(f"Unsupported ablation: {self.ablation}")

        expected = self._expected_dim()
        if X.shape[1] != expected:
            raise RuntimeError(
                f"Ablation '{self.ablation}' produced shape {X.shape[1]}, expected {expected}"
            )

        return X, y

    def align_and_split(
        self, v: np.ndarray, i: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self._build_features(v, i, t)
        self.X_all = X
        self.y_all = y

        self.v_aligned = np.asarray(v[1:], dtype=np.float64)
        self.i_aligned = np.asarray(i[1:], dtype=np.float64)
        self.t_aligned = np.asarray(t[1:], dtype=np.float64)

        n = len(y)
        train_r = float(getattr(config, "TRAIN_RATIO", 0.4))
        cal_r = float(getattr(config, "CAL_RATIO", 0.2))
        val_r = float(getattr(config, "VAL_RATIO", 0.2))

        n_train = int(n * train_r)
        n_cal = int(n * cal_r)
        n_val = int(n * val_r)

        self.train_idx = (0, n_train)
        self.cal_idx = (n_train, n_train + n_cal)
        self.val_idx = (n_train + n_cal, n_train + n_cal + n_val)
        self.test_idx = (n_train + n_cal + n_val, n)

        return self.X_all, self.y_all

    def get_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        if split == "train":
            start, end = self.train_idx
        elif split == "cal":
            start, end = self.cal_idx
        elif split == "val":
            start, end = self.val_idx
        elif split == "test":
            start, end = self.test_idx
        else:
            raise ValueError(f"Unknown split: '{split}'")

        return self.X_all[start:end], self.y_all[start:end]

    def _extract_physical_inputs(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        is_std = False
        if self.feature_mean is not None and self.feature_std is not None:
            if len(self.feature_mean) == X.shape[1] and self.feature_mean[0] > 10.0:
                is_std = bool(abs(float(np.mean(X[:, 0]))) < 5.0)

        if is_std:
            std_safe = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
            X_raw = X * std_safe + self.feature_mean
            X_norm = X
        else:
            X_raw = X
            if self.feature_mean is not None and self.feature_std is not None and len(self.feature_mean) == X.shape[1]:
                std_safe = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
                X_norm = (X - self.feature_mean) / std_safe
            else:
                mean_loc = np.mean(X, axis=0)
                std_loc = np.std(X, axis=0)
                std_loc = np.where(std_loc < 1e-8, 1.0, std_loc)
                X_norm = (X - mean_loc) / std_loc

        t_prev = X_raw[:, 0]

        if self.ablation in ("full_cross_modal", "no_temporal_lag"):
            i_prev = X_raw[:, 2]
            i_eff_sq = i_prev**2
        elif self.ablation == "missing_voltage":
            i_prev = X_raw[:, 1]
            i_eff_sq = i_prev**2
        elif self.ablation == "missing_current":
            v_prev = X_raw[:, 1]
            i_est = v_prev / 12.5
            i_eff_sq = i_est**2
        elif self.ablation == "single_modal_ar":
            i_eff_sq = np.zeros_like(t_prev)
        else:
            i_eff_sq = np.zeros_like(t_prev)

        return t_prev, i_eff_sq, X_norm

    def fit(self) -> "CrossModalGate":
        X_train, y_train = self.get_split("train")
        t_prev, i_eff_sq, X_norm = self._extract_physical_inputs(X_train)
        delta_y = y_train - t_prev
        n = len(y_train)

        # 1. Fit thermodynamic ODE on unclipped clean telemetry
        valid_mask = (t_prev < (self.t_clip_max - 2.0)) & (y_train < (self.t_clip_max - 2.0))
        if np.sum(valid_mask) < 50:
            valid_mask = np.ones(n, dtype=bool)

        t_fit = t_prev[valid_mask]
        i_fit = i_eff_sq[valid_mask]
        dy_fit = delta_y[valid_mask]

        if self.ablation == "single_modal_ar":
            M = np.column_stack([t_fit, np.ones(len(t_fit), dtype=np.float64)])
            theta = np.linalg.lstsq(M, dy_fit, rcond=None)[0]
            self.beta_1 = 0.0
            self.beta_2 = float(min(-1e-4, theta[0]))
            self.beta_0 = float(theta[1])
        else:
            M = np.column_stack([i_fit, t_fit, np.ones(len(t_fit), dtype=np.float64)])
            theta = np.linalg.lstsq(M, dy_fit, rcond=None)[0]
            self.beta_1 = float(max(1e-5, theta[0]))
            self.beta_2 = float(min(-1e-4, theta[1]))
            self.beta_0 = float(theta[2])

        pred_delta_raw = self.beta_1 * i_fit + self.beta_2 * t_fit + self.beta_0
        pred_t = np.clip(t_fit + pred_delta_raw, self.t_clip_min, self.t_clip_max)
        e_fit = dy_fit - (pred_t - t_fit)
        dyn_std = float(np.std(e_fit))
        self.dyn_scale = dyn_std if dyn_std > 1e-4 else 0.05

        # 2. Fit ablation-specific Ridge model on all available features
        X_b = np.hstack([X_norm, np.ones((X_norm.shape[0], 1), dtype=np.float64)])
        d = X_b.shape[1]
        reg = self.ridge_alpha * np.eye(d, dtype=np.float64)
        reg[-1, -1] = 0.0

        A = X_b.T @ X_b + reg
        b = X_b.T @ delta_y
        self.weights = np.linalg.solve(A, b)

        stat_delta_pred = X_b @ self.weights
        stat_residuals = np.abs(delta_y - stat_delta_pred)
        stat_std = float(np.std(stat_residuals))
        self.stat_scale = stat_std if stat_std > 1e-4 else 0.05

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t_prev, i_eff_sq, _ = self._extract_physical_inputs(X)
        pred_delta_raw = self.beta_1 * i_eff_sq + self.beta_2 * t_prev + self.beta_0
        return np.clip(t_prev + pred_delta_raw, self.t_clip_min, self.t_clip_max)

    def physical_residual(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        t_pred = self.predict(X)
        return np.abs(y - t_pred)

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        t_prev, i_eff_sq, X_norm = self._extract_physical_inputs(X)
        delta_y = y - t_prev
        n = len(delta_y)

        # 1. Physics ODE error under exact thermal constraints
        pred_delta_raw = self.beta_1 * i_eff_sq + self.beta_2 * t_prev + self.beta_0
        pred_t = np.clip(t_prev + pred_delta_raw, self.t_clip_min, self.t_clip_max)
        e_phys = delta_y - (pred_t - t_prev)
        s_phys = np.abs(e_phys) / self.dyn_scale

        # 2. Rolling boxcar variance normalization (sensitive to static bias injection)
        w = min(self.window_size, n)
        cumsum = np.cumsum(np.insert(e_phys, 0, 0.0))
        rolling_sum = np.empty(n, dtype=np.float64)
        counts = np.empty(n, dtype=np.float64)

        for k in range(w):
            rolling_sum[k] = cumsum[k + 1]
            counts[k] = float(k + 1)

        rolling_sum[w:] = cumsum[w + 1 :] - cumsum[1 : n - w + 1]
        counts[w:] = float(w)

        s_window = np.abs(rolling_sum / counts) / (self.dyn_scale / np.sqrt(counts))
        s_primary = np.maximum(s_phys, s_window)

        if self.ablation == "single_modal_ar" or self.weights is None or n == 0:
            return s_primary

        # 3. Soft-bounded statistical cross-check to differentiate feature ablations
        # Capping the standardized statistical residual at 2.5 avoids inflating tau during regime transitions
        X_b = np.hstack([X_norm, np.ones((X_norm.shape[0], 1), dtype=np.float64)])
        e_stat = delta_y - (X_b @ self.weights)
        s_stat_bounded = np.clip(np.abs(e_stat) / self.stat_scale, 0.0, 2.5)

        return s_primary + 0.10 * s_stat_bounded

    def calibrate(self, alpha: Optional[float] = None) -> "CrossModalGate":
        if alpha is None:
            alpha = float(getattr(config, "ALPHA", 0.05))
        else:
            alpha = float(alpha)

        X_cal, y_cal = self.get_split("cal")
        cal_scores = self.score(X_cal, y_cal)

        n = len(cal_scores)
        sorted_scores = np.sort(cal_scores)
        rank = int(np.ceil((n + 1) * (1.0 - alpha)))
        rank = min(n, max(1, rank))
        self.threshold = float(sorted_scores[rank - 1])
        self.calibrated = True
        return self

    def verify_arrays(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        if not self.calibrated or self.threshold is None:
            raise RuntimeError("Gate has not been calibrated. Call calibrate() first.")

        scores = self.score(X, y)
        threshold = float(self.threshold)
        anomaly_mask = scores > threshold
        keep_mask = ~anomaly_mask

        return scores, threshold, anomaly_mask, keep_mask

    def verify(
        self, split: str = "test"
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        X, y = self.get_split(split)
        return self.verify_arrays(X, y)