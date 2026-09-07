"""
verify_contracts.py
Immediate pre-flight validation of all crossmodal_gate, fl_engine, 
and run_experiments input/output contracts.
"""

import inspect
import sys
from collections import OrderedDict

import numpy as np
import torch

import config
from crossmodal_gate import CrossModalGate
import fl_engine


def make_fl_model(ablation: str, dim: int):
    sig = inspect.signature(fl_engine.FLModel.__init__)
    params = set(sig.parameters.keys())

    if "ablation" in params:
        return fl_engine.FLModel(ablation=ablation)
    elif "input_dim" in params:
        return fl_engine.FLModel(input_dim=dim)
    elif "in_features" in params:
        return fl_engine.FLModel(in_features=dim)
    elif "dim" in params:
        return fl_engine.FLModel(dim=dim)
    elif "input_size" in params:
        return fl_engine.FLModel(input_size=dim)
    else:
        try:
            return fl_engine.FLModel(ablation)
        except Exception:
            return fl_engine.FLModel(dim)


def run_contract_checks():
    print("=" * 60)
    print("STARTING FULL PRE-FLIGHT CONTRACT VERIFICATION")
    print("=" * 60)

    n_samples = 1000
    np.random.seed(42)
    v = 220.0 + np.random.randn(n_samples) * 0.5
    i = 17.6 + np.random.randn(n_samples) * 0.1
    t = 25.0 + np.cumsum(np.random.randn(n_samples) * 0.05)

    expected_dims = {
        "full_cross_modal": 6,
        "missing_current": 4,
        "missing_voltage": 4,
        "no_temporal_lag": 4,
        "single_modal_ar": 2,
    }

    print("\n[1/5] Checking Ablation Feature Dimensions & Splits:")
    for ablation, exp_dim in expected_dims.items():
        gate = CrossModalGate(ablation=ablation)
        ret = gate.align_and_split(v, i, t)
        assert isinstance(ret, tuple) and len(ret) == 2, "align_and_split must return (X, y)"
        X_train, y_train = gate.get_split("train")

        assert X_train.ndim == 2, f"{ablation}: X_train must be 2D, got {X_train.ndim}D"
        assert (
            X_train.shape[1] == exp_dim
        ), f"{ablation}: Expected dimension {exp_dim}, got {X_train.shape[1]}"
        assert len(X_train) == len(
            y_train
        ), f"{ablation}: X and y length mismatch ({len(X_train)} vs {len(y_train)})"
        print(f"  ✓ {ablation:<20} -> Shape: {X_train.shape} (Dim: {exp_dim})")

    print("\n[2/5] Checking Index Boundary Contracts ((start, end) tuples):")
    gate = CrossModalGate(ablation="full_cross_modal")
    gate.align_and_split(v, i, t)

    for split_name, idx_tuple in [
        ("train_idx", gate.train_idx),
        ("cal_idx", gate.cal_idx),
        ("val_idx", gate.val_idx),
        ("test_idx", gate.test_idx),
    ]:
        assert isinstance(
            idx_tuple, tuple
        ), f"{split_name} must be a tuple, got {type(idx_tuple)}"
        assert len(idx_tuple) == 2, f"{split_name} must have length 2, got {len(idx_tuple)}"
        start, end = idx_tuple
        assert isinstance(start, (int, np.integer)), f"{split_name} start must be integer"
        assert isinstance(end, (int, np.integer)), f"{split_name} end must be integer"
        assert 0 <= start < end <= (n_samples - 1), f"Invalid range ({start}, {end})"

        slice_test = v[start : end + 1]
        assert len(slice_test) == (end - start + 1), f"Slice mismatch for {split_name}"
        print(f"  ✓ {split_name:<12} -> ({start}, {end}) [Length: {end - start}] unpacked successfully")

    print("\n[3/5] Checking Gate Verification 4-Tuple Outputs & Conformal FPR:")
    gate.fit()
    gate.calibrate(alpha=0.05)

    res_val = gate.verify("val")
    assert isinstance(res_val, tuple), f"verify() must return a tuple, got {type(res_val)}"
    assert len(res_val) == 4, f"verify() must return 4 elements, got {len(res_val)}"

    scores, threshold, anomaly_mask, keep_mask = res_val
    assert isinstance(scores, np.ndarray), f"scores must be ndarray, got {type(scores)}"
    assert isinstance(threshold, (float, np.floating)), f"threshold must be float"
    assert isinstance(anomaly_mask, np.ndarray), f"anomaly_mask must be ndarray"
    assert anomaly_mask.dtype == bool, f"anomaly_mask must be bool"
    assert isinstance(keep_mask, np.ndarray), f"keep_mask must be ndarray"
    assert keep_mask.dtype == bool, f"keep_mask must be bool"
    assert np.all(anomaly_mask == ~keep_mask), "keep_mask must be bitwise NOT of anomaly_mask"

    emp_fpr = float(anomaly_mask.mean())
    assert (
        0.01 <= emp_fpr <= 0.12
    ), f"Empirical FPR {emp_fpr:.4f} deviated significantly from nominal alpha=0.05"
    print(f"  ✓ verify('val')       -> 4 items unpacked. Empirical FPR = {emp_fpr:.4f} (Nominal 0.05)")

    X_test, y_test = gate.get_split("test")
    res_arr = gate.verify_arrays(X_test, y_test)
    assert len(res_arr) == 4, "verify_arrays must return 4 elements"
    s2, th2, anom2, keep2 = res_arr
    print(f"  ✓ verify_arrays(X, y) -> 4 items unpacked. Empirical FPR = {anom2.mean():.4f}")

    print("\n[4/5] Checking FLModel Input/Output Tensor Passes:")
    for ablation, dim in expected_dims.items():
        model = make_fl_model(ablation=ablation, dim=dim)
        batch = torch.randn(16, dim)
        out = model(batch)
        assert out.shape == (
            16,
            1,
        ), f"FLModel({ablation}) forward returned shape {out.shape}, expected (16, 1)"
        print(f"  ✓ FLModel for {ablation:<18} (dim={dim}) -> Forward pass shape: {out.shape}")

    print("\n[5/5] Checking FL Training, Aggregation & Delta Contracts:")
    model = make_fl_model(ablation="full_cross_modal", dim=6)
    X_train, y_train = gate.get_split("train")
    keep_mask = np.ones(len(y_train), dtype=bool)

    delta = fl_engine.train_local_model(
        model.state_dict(),
        X_train,
        y_train,
        keep_mask=keep_mask,
        ablation="full_cross_modal",
        local_epochs=1,
        learning_rate=0.01,
        return_delta=True,
    )
    assert isinstance(delta, (dict, OrderedDict)), "train_local_model must return delta dict"
    print("  ✓ train_local_model executed successfully with return_delta=True")

    # Invertibility check: unflatten(flatten(w)) == w
    vec = fl_engine.flatten_state_dict(model.state_dict())
    reconstructed = fl_engine.unflatten_vector(vec, model.state_dict())
    for k in model.state_dict():
        diff = (model.state_dict()[k].float() - reconstructed[k].float()).abs().max().item()
        assert diff < 1e-6, f"Weight mismatch on unflatten for {k}: {diff}"
    print("  ✓ flatten_state_dict & unflatten_vector round-trip passed (< 1e-6 error)")

    # Aggregators
    updates = [delta, delta, delta]
    for method in ["fedavg", "coordinate_median", "trimmed_mean", "krum"]:
        agg = fl_engine.aggregate_updates(updates, aggregator=method)
        assert isinstance(agg, (dict, OrderedDict)), f"{method} returned wrong type"
        print(f"  ✓ Aggregation [{method:<18}] -> Verified")

    # Delta application
    base_model = make_fl_model(ablation="full_cross_modal", dim=6)
    fl_engine.apply_delta(base_model, delta)
    print("  ✓ apply_delta executed successfully")

    # Physical MSE evaluation
    mse_val = fl_engine.evaluate_global_model_physical_mse(
        model.state_dict(), X_test, y_test, ablation="full_cross_modal"
    )
    assert np.isfinite(mse_val), "MSE evaluation returned non-finite value"
    print(f"  ✓ evaluate_global_model_physical_mse -> Verified (MSE = {float(mse_val):.6f})")

    print("\n" + "=" * 60)
    print("ALL API AND CONTRACT CHECKS PASSED CONVERTIBLY.")
    print("=" * 60)


if __name__ == "__main__":
    run_contract_checks()