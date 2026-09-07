"""
fl_engine.py
Federated learning engine for finite-sample conformal cross-modal verification.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FLModel(nn.Module):
    """
    Deterministic regression model used by the FL pipeline.
    Input dimension is determined by the configured ablation.
    """

    FEATURE_DIMS = {
        "full_cross_modal": 6,
        "missing_current": 4,
        "missing_voltage": 4,
        "no_temporal_lag": 4,
        "single_modal_ar": 2,
    }

    def __init__(
        self,
        ablation: Union[str, int] = "full_cross_modal",
        input_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if isinstance(ablation, int):
            input_dim = ablation
            self.ablation = "custom"
        elif input_dim is not None:
            self.ablation = str(ablation)
        else:
            if ablation not in self.FEATURE_DIMS:
                raise ValueError(
                    f"Unknown ablation '{ablation}'. "
                    f"Expected one of {sorted(self.FEATURE_DIMS)}."
                )
            self.ablation = ablation
            input_dim = self.FEATURE_DIMS[ablation]

        hidden_dim = 32

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _clone_state(
    state: Dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((k, v.detach().clone()) for k, v in state.items())


def flatten_state_dict(
    state_dict: Dict[str, torch.Tensor],
    keys: Optional[Iterable[str]] = None,
) -> torch.Tensor:
    """
    Flatten a state dictionary into a single 1-D tensor.
    Strictly follows insertion order unless explicit keys are provided.
    """
    if keys is None:
        keys = list(state_dict.keys())

    chunks = []
    for key in keys:
        tensor = state_dict[key]
        if not torch.is_floating_point(tensor):
            continue
        chunks.append(tensor.detach().reshape(-1))

    if not chunks:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat(chunks)


def flatten_vector(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    return flatten_state_dict(state_dict)


def unflatten_vector(
    vector: Union[torch.Tensor, np.ndarray],
    template: Dict[str, torch.Tensor],
    keys: Optional[Iterable[str]] = None,
) -> OrderedDict[str, torch.Tensor]:
    """
    Reconstruct a state-dict from a flattened 1-D vector.
    Consumes vector entries in identical key order as flatten_state_dict.
    """
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32)

    vector = vector.detach().reshape(-1)
    output = OrderedDict()

    if keys is None:
        keys = list(template.keys())

    offset = 0
    for key in keys:
        tensor = template[key]
        if not torch.is_floating_point(tensor):
            output[key] = tensor.detach().clone()
            continue

        numel = tensor.numel()
        end = offset + numel

        if end > len(vector):
            raise ValueError(
                f"Flattened vector is too short to reconstruct state entry '{key}'."
            )

        output[key] = (
            vector[offset:end].reshape(tensor.shape).to(dtype=tensor.dtype)
        )
        offset = end

    if offset != len(vector):
        raise ValueError(
            f"Flattened vector contains unused entries: {len(vector) - offset}."
        )

    return output


def apply_delta(
    global_state: Union[Dict[str, torch.Tensor], nn.Module],
    delta: Union[Dict[str, torch.Tensor], torch.Tensor, np.ndarray],
) -> OrderedDict[str, torch.Tensor]:
    is_module = isinstance(global_state, nn.Module)
    state_dict = global_state.state_dict() if is_module else global_state

    if not isinstance(delta, (dict, OrderedDict)):
        delta = unflatten_vector(delta, state_dict)

    if set(state_dict.keys()) != set(delta.keys()):
        missing = set(state_dict.keys()) - set(delta.keys())
        extra = set(delta.keys()) - set(state_dict.keys())
        raise ValueError(
            f"Key mismatch in apply_delta. Missing={sorted(missing)}, Extra={sorted(extra)}."
        )

    new_state = OrderedDict()
    for key, global_tensor in state_dict.items():
        delta_tensor = delta[key]
        if torch.is_floating_point(global_tensor):
            updated = (
                global_tensor.detach()
                + delta_tensor.detach().to(global_tensor.dtype)
            )
            if not torch.isfinite(updated).all():
                raise FloatingPointError(
                    f"Non-finite parameter after applying delta to '{key}'."
                )
            new_state[key] = updated
        else:
            new_state[key] = global_tensor.detach().clone()

    if is_module:
        global_state.load_state_dict(new_state)

    return new_state


def _model_from_state(
    global_state: Dict[str, torch.Tensor],
    ablation: str,
) -> FLModel:
    model = FLModel(ablation)
    current_state = model.state_dict()

    if set(current_state.keys()) != set(global_state.keys()):
        raise ValueError("Global state is incompatible with FLModel.")

    cleaned_state = OrderedDict()
    for key, current_tensor in current_state.items():
        source = global_state[key]
        if source.shape != current_tensor.shape:
            raise ValueError(
                f"Shape mismatch for '{key}': "
                f"global={tuple(source.shape)}, "
                f"model={tuple(current_tensor.shape)}."
            )
        cleaned_state[key] = source.detach().clone().to(dtype=current_tensor.dtype)

    model.load_state_dict(cleaned_state, strict=True)
    return model


def train_local_model(
    global_state: Union[Dict[str, torch.Tensor], nn.Module],
    X: Union[np.ndarray, torch.Tensor, DataLoader],
    y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    keep_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    local_epochs: int = 1,
    learning_rate: float = 0.01,
    seed: int = 42,
    ablation: str = "full_cross_modal",
    batch_size: int = 32,
    gradient_clip_norm: float = 10.0,
    return_delta: bool = False,
    **kwargs,
) -> Union[OrderedDict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if "epochs" in kwargs:
        local_epochs = kwargs["epochs"]
    if "lr" in kwargs:
        learning_rate = kwargs["lr"]

    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if isinstance(global_state, nn.Module):
        state_dict_ref = OrderedDict(
            (k, v.detach().clone()) for k, v in global_state.state_dict().items()
        )
        if hasattr(global_state, "ablation"):
            ablation = global_state.ablation
    else:
        state_dict_ref = global_state

    if y is None and isinstance(X, DataLoader):
        all_X, all_y = [], []
        for bx, by in X:
            all_X.append(bx.cpu().numpy() if isinstance(bx, torch.Tensor) else bx)
            all_y.append(by.cpu().numpy() if isinstance(by, torch.Tensor) else by)
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2-D feature matrix.")
    if len(X) != len(y):
        raise ValueError("X and y lengths differ.")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values.")
    if not np.isfinite(y).all():
        raise ValueError("y contains non-finite values.")

    expected_dim = FLModel.FEATURE_DIMS.get(ablation, X.shape[1])
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch for ablation '{ablation}': "
            f"got {X.shape[1]}, expected {expected_dim}."
        )

    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
        if len(keep_mask) != len(X):
            raise ValueError("keep_mask length differs from training data.")
        X = X[keep_mask]
        y = y[keep_mask]

    if len(X) == 0:
        if return_delta:
            return OrderedDict(
                (k, torch.zeros_like(v) if torch.is_floating_point(v) else v.detach().clone())
                for k, v in state_dict_ref.items()
            )
        return _clone_state(state_dict_ref)

    model = _model_from_state(state_dict_ref, ablation)
    model.train()

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    generator = torch.Generator()
    generator.manual_seed(seed)

    try:
        import config
        cfg_batch = int(getattr(config, "BATCH_SIZE", batch_size))
        cfg_clip = float(getattr(config, "GRADIENT_CLIP_NORM", gradient_clip_norm))
    except Exception:
        cfg_batch = int(batch_size)
        cfg_clip = float(gradient_clip_norm)

    loader = DataLoader(
        dataset,
        batch_size=max(1, cfg_batch),
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()

    for _ in range(int(local_epochs)):
        for batch_X, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_X).reshape(-1)
            loss = criterion(prediction, batch_y)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite local training loss.")

            loss.backward()
            if cfg_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max(float(cfg_clip), 1e-12),
                )
            optimizer.step()

    local_state = OrderedDict(
        (k, v.detach().clone()) for k, v in model.state_dict().items()
    )

    if not return_delta:
        return local_state

    delta = OrderedDict()
    for key, global_tensor in state_dict_ref.items():
        local_tensor = local_state[key]
        if torch.is_floating_point(global_tensor):
            delta[key] = local_tensor - global_tensor.detach()
        else:
            delta[key] = global_tensor.detach().clone()

    return delta


def _stack_delta_vectors(
    updates: Iterable[Dict[str, torch.Tensor]],
) -> Tuple[list[str], torch.Tensor]:
    if len(updates) == 0:
        raise ValueError("Cannot aggregate an empty update list.")

    keys = list(updates[0].keys())
    vectors = []

    for update in updates:
        if set(update.keys()) != set(keys):
            raise ValueError("All updates must contain identical keys.")
        vectors.append(flatten_state_dict(update, keys=keys))

    return keys, torch.stack(vectors, dim=0)


def _weighted_mean(
    matrix: torch.Tensor,
    weights: Optional[Union[Iterable[float], torch.Tensor, np.ndarray]] = None,
) -> torch.Tensor:
    n = matrix.shape[0]
    if weights is None:
        return torch.mean(matrix, dim=0)

    weights = torch.tensor(
        np.asarray(weights, dtype=np.float64),
        dtype=matrix.dtype,
    )

    if len(weights) != n:
        raise ValueError("sample_counts length differs from update count.")
    if torch.any(weights < 0):
        raise ValueError("sample_counts cannot be negative.")

    total = torch.sum(weights)
    if float(total) <= 0:
        return torch.mean(matrix, dim=0)

    normalized = weights / total
    return torch.sum(matrix * normalized.reshape(-1, 1), dim=0)


def _coordinate_median(matrix: torch.Tensor) -> torch.Tensor:
    return torch.median(matrix, dim=0).values


def _trimmed_mean(matrix: torch.Tensor, trim_fraction: float) -> torch.Tensor:
    n = matrix.shape[0]
    trim = int(np.floor(n * trim_fraction))
    if 2 * trim >= n:
        return torch.mean(matrix, dim=0)

    sorted_values, _ = torch.sort(matrix, dim=0)
    trimmed = sorted_values[trim : n - trim]
    return torch.mean(trimmed, dim=0)


def _krum(matrix: torch.Tensor, f: int) -> torch.Tensor:
    n = matrix.shape[0]
    f = int(f)

    if n < 2 * f + 3:
        raise ValueError(f"Krum requires N >= 2f + 3. Got N={n}, f={f}.")

    distances = torch.cdist(matrix, matrix, p=2)
    scores = []
    neighbor_count = n - f - 2

    for i in range(n):
        row = distances[i]
        row_without_self = torch.cat([row[:i], row[i + 1 :]])
        nearest = torch.topk(row_without_self, k=neighbor_count, largest=False).values
        scores.append(torch.sum(nearest * nearest))

    selected = int(torch.argmin(torch.stack(scores)).item())
    return matrix[selected]


def aggregate_updates(
    updates: Union[list[Dict[str, torch.Tensor]], list[torch.Tensor], torch.Tensor],
    aggregator: str = "fedavg",
    f: int = 0,
    sample_counts: Optional[Iterable[float]] = None,
    method: Optional[str] = None,
    **kwargs,
) -> Union[OrderedDict[str, torch.Tensor], torch.Tensor]:
    if not updates:
        raise ValueError("No client updates supplied.")

    if method is not None:
        aggregator = method

    aggregator = str(aggregator).lower()

    if isinstance(updates[0], (dict, OrderedDict)):
        is_dict = True
        template = updates[0]
        keys, matrix = _stack_delta_vectors(updates)
    else:
        is_dict = False
        template = None
        keys = None
        if isinstance(updates, torch.Tensor) and updates.ndim == 2:
            matrix = updates
        else:
            matrix = torch.stack(
                [torch.as_tensor(u, dtype=torch.float32) for u in updates],
                dim=0,
            )

    n = matrix.shape[0]

    if aggregator == "fedavg":
        vector = _weighted_mean(matrix, sample_counts)
    elif aggregator == "coordinate_median":
        vector = _coordinate_median(matrix)
    elif aggregator == "trimmed_mean":
        if n <= 1:
            vector = matrix[0]
        else:
            trim_fraction = float(f) / float(n)
            vector = _trimmed_mean(matrix, trim_fraction)
    elif aggregator == "krum":
        vector = _krum(matrix, int(f))
    else:
        raise ValueError(
            f"Unknown aggregator '{aggregator}'. "
            "Expected fedavg, coordinate_median, trimmed_mean, or krum."
        )

    if is_dict:
        return unflatten_vector(vector, template, keys=keys)

    return vector


def evaluate_global_model_physical_mse(
    global_state: Union[Dict[str, torch.Tensor], nn.Module],
    X_test: Union[np.ndarray, torch.Tensor, DataLoader],
    y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ablation: str = "full_cross_modal",
    target_std: float = 1.0,
) -> float:
    if isinstance(global_state, nn.Module):
        model = global_state
    else:
        model = _model_from_state(global_state, ablation)

    model.eval()

    if y_test is None and isinstance(X_test, DataLoader):
        all_X, all_y = [], []
        for bx, by in X_test:
            all_X.append(bx.cpu().numpy() if isinstance(bx, torch.Tensor) else bx)
            all_y.append(by.cpu().numpy() if isinstance(by, torch.Tensor) else by)
        X_test = np.concatenate(all_X, axis=0)
        y_test = np.concatenate(all_y, axis=0)

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).reshape(-1)

    if X_test.ndim != 2:
        raise ValueError("X_test must be a 2-D array.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test lengths differ.")
    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains non-finite values.")
    if not np.isfinite(y_test).all():
        raise ValueError("y_test contains non-finite values.")

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test)
        prediction = model(X_tensor).reshape(-1).cpu().numpy()

    if not np.isfinite(prediction).all():
        raise FloatingPointError("Global model produced non-finite predictions.")

    scale = max(abs(float(target_std)), 1e-12)
    residual = (prediction - y_test) * scale
    mse = float(np.mean(residual * residual))

    if not np.isfinite(mse):
        raise FloatingPointError("Physical MSE is non-finite.")

    return mse