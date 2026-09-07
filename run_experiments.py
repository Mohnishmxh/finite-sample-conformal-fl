"""
run_experiments.py
------------------
Final reproducible research pipeline.

Experiments:

E1 - Clean validation false-positive rate
E2 - Tier-1 physical attack detection
E3 - Multi-round FL under Tier-1 poisoning
E4 - Multi-round Byzantine aggregation robustness
E5 - Physical heterogeneity impact
E6 - Attack-balanced ablation study
E7 - Software overhead profiling
"""

import json
import os
import random
import time

import numpy as np
import torch

import config

from physics_simulator import (
    MotorPhysicsSimulator,
)

from crossmodal_gate import (
    CrossModalGate,
)

from fl_engine import (
    FLModel,
    train_local_model,
    aggregate_updates,
    evaluate_global_model_physical_mse,
    flatten_state_dict,
    unflatten_vector,
    apply_delta,
)

from attacks import (
    apply_tier1_attack,
    apply_tier2_byzantine,
)

from profiler import (
    profile_execution,
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def seed_everything(seed):
    seed = int(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    if config.DETERMINISTIC_TORCH:

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# METRICS
# =========================================================

def compute_classification_metrics(
    anomaly_mask,
    y_true,
):
    anomaly_mask = np.asarray(
        anomaly_mask,
        dtype=bool,
    )

    y_true = np.asarray(
        y_true,
        dtype=bool,
    )

    if len(anomaly_mask) != len(
        y_true
    ):
        raise ValueError(
            "Prediction and ground truth lengths differ."
        )

    TP = int(
        np.sum(
            anomaly_mask
            & y_true
        )
    )

    FP = int(
        np.sum(
            anomaly_mask
            & ~y_true
        )
    )

    TN = int(
        np.sum(
            ~anomaly_mask
            & ~y_true
        )
    )

    FN = int(
        np.sum(
            ~anomaly_mask
            & y_true
        )
    )

    precision = (
        TP
        / max(
            1,
            TP + FP,
        )
    )

    recall = (
        TP
        / max(
            1,
            TP + FN,
        )
    )

    if precision + recall > 0:

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    fpr = (
        FP
        / max(
            1,
            FP + TN,
        )
    )

    return {
        "tp": TP,
        "fp": FP,
        "tn": TN,
        "fn": FN,
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "fpr": float(
            fpr
        ),
    }


# =========================================================
# RAW ALIGNMENT
# =========================================================

def aligned_raw_segment(
    v,
    i,
    t,
    aligned_start,
    aligned_end,
):
    """
    Aligned j uses raw j and j+1.

    Therefore aligned [s:e] requires raw [s:e+1].
    """

    return (
        v[
            aligned_start:
            aligned_end + 1
        ].copy(),

        i[
            aligned_start:
            aligned_end + 1
        ].copy(),

        t[
            aligned_start:
            aligned_end + 1
        ].copy(),
    )


def build_raw_aligned(
    v,
    i,
    t,
    ablation="full_cross_modal",
):
    gate = CrossModalGate(
        ablation=ablation
    )

    X, y = gate.align_and_split(
        v,
        i,
        t,
    )

    return (
        X.copy(),
        y.copy(),
    )


# =========================================================
# TRAINING-ONLY GLOBAL STANDARDIZER
# =========================================================

def fit_global_standardizer(
    clients_raw,
    ablation="full_cross_modal",
):
    """
    Fit ONE global standardizer using only aligned training
    samples from all clients.

    This avoids client-specific coefficient spaces and keeps
    FL aggregation mathematically compatible.
    """

    X_parts = []
    y_parts = []

    for v, i, t in clients_raw:

        gate = CrossModalGate(
            ablation=ablation
        )

        gate.align_and_split(
            v,
            i,
            t,
        )

        X_train, y_train = (
            gate.get_split(
                "train"
            )
        )

        X_parts.append(
            X_train
        )

        y_parts.append(
            y_train
        )

    X_train_all = np.vstack(
        X_parts
    )

    y_train_all = np.concatenate(
        y_parts
    )

    feature_mean = np.mean(
        X_train_all,
        axis=0,
    )

    feature_std = np.std(
        X_train_all,
        axis=0,
    )

    feature_std = np.maximum(
        feature_std,
        config.NORMALIZATION_EPS,
    )

    target_mean = float(
        np.mean(
            y_train_all
        )
    )

    target_std = float(
        np.std(
            y_train_all
        )
    )

    target_std = max(
        target_std,
        config.NORMALIZATION_EPS,
    )

    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def standardize_arrays(
    X,
    y,
    scaler,
):
    X_norm = (
        np.asarray(
            X,
            dtype=float,
        )
        - scaler["feature_mean"]
    ) / (
        scaler["feature_std"]
        + config.NORMALIZATION_EPS
    )

    y_norm = (
        np.asarray(
            y,
            dtype=float,
        )
        - scaler["target_mean"]
    ) / (
        scaler["target_std"]
        + config.NORMALIZATION_EPS
    )

    return (
        X_norm,
        y_norm,
    )


# =========================================================
# GATE CONSTRUCTION
# =========================================================

def fit_gate(
    v,
    i,
    t,
    ablation,
    scaler,
):
    gate = CrossModalGate(
        ablation=ablation,
        feature_mean=scaler[
            "feature_mean"
        ],
        feature_std=scaler[
            "feature_std"
        ],
        target_mean=scaler[
            "target_mean"
        ],
        target_std=scaler[
            "target_std"
        ],
    )

    gate.align_and_split(
        v,
        i,
        t,
    )

    gate.fit()

    gate.calibrate(
        alpha=config.ALPHA
    )

    return gate


# =========================================================
# PRE-EXECUTION AUDIT
# =========================================================

def pre_execution_audit():

    ratio_sum = (
        config.TRAIN_RATIO
        + config.CAL_RATIO
        + config.VAL_RATIO
        + config.TEST_RATIO
    )

    assert np.isclose(
        ratio_sum,
        1.0,
    )

    M = (
        config.NUM_SAMPLES
        - 1
    )

    assert M > 0

    f = int(
        config.BYZANTINE_F
    )

    assert (
        config.NUM_CLIENTS
        >= 2 * f + 1
    )

    assert (
        config.NUM_CLIENTS
        >= 2 * f + 3
    )

    sim = MotorPhysicsSimulator(
        seed=42,
        heterogeneity_scale=1.0,
    )

    v, i, t = (
        sim.generate_telemetry(
            200
        )
    )

    assert np.isfinite(v).all()
    assert np.isfinite(i).all()
    assert np.isfinite(t).all()

    assert np.var(t) > 0

    print(
        "[PASS] Pre-execution audit complete."
    )

    print(
        f"[INFO] Raw samples: {config.NUM_SAMPLES}"
    )

    print(
        f"[INFO] Aligned samples: {M}"
    )

    print(
        f"[INFO] Clients: {config.NUM_CLIENTS}"
    )

    print(
        f"[INFO] FL rounds: {config.NUM_ROUNDS}"
    )

    print(
        f"[INFO] Byzantine f: {f}"
    )


# =========================================================
# CLIENT DATA
# =========================================================

def prepare_client_data(
    clients_raw,
    scaler,
    ablation="full_cross_modal",
):
    client_data = []

    for v, i, t in clients_raw:

        gate = fit_gate(
            v,
            i,
            t,
            ablation,
            scaler,
        )

        X_train, y_train = (
            gate.get_split(
                "train"
            )
        )

        X_test, y_test = (
            gate.get_split(
                "test"
            )
        )

        X_train_norm, y_train_norm = (
            standardize_arrays(
                X_train,
                y_train,
                scaler,
            )
        )

        X_test_norm, y_test_norm = (
            standardize_arrays(
                X_test,
                y_test,
                scaler,
            )
        )

        client_data.append(
            {
                "gate": gate,
                "X_train": X_train_norm,
                "y_train": y_train_norm,
                "X_test": X_test_norm,
                "y_test": y_test_norm,
                "n_train": len(
                    X_train_norm
                ),
            }
        )

    return client_data


# =========================================================
# MULTI-ROUND FEDERATED TRAINING
# =========================================================

def run_federated_training(
    client_data,
    seed,
    poisoned_clients=None,
    defended=False,
    aggregator="fedavg",
    tier2_attack=None,
):
    """
    Run config.NUM_ROUNDS rounds.

    For Tier-1 poisoning:
        poisoned clients use corrupted training data.

    For defended mode:
        the clean gate is used to generate a keep mask and
        anomalous training samples are removed.

    For Tier-2:
        attacks are applied to complete flattened deltas.
    """

    if poisoned_clients is None:
        poisoned_clients = set()

    N = len(
        client_data
    )

    f = config.BYZANTINE_F

    seed_everything(
        seed
    )

    global_model = FLModel(
        "full_cross_modal"
    )

    global_state = {
        key: value.detach().clone()
        for key, value
        in global_model.state_dict().items()
    }

    for round_idx in range(
        config.NUM_ROUNDS
    ):

        round_deltas = []
        sample_counts = []

        for client_id in range(N):

            data = client_data[
                client_id
            ]

            X_train = data[
                "X_train"
            ]

            y_train = data[
                "y_train"
            ]

            keep_mask = None

            # -------------------------------------------------
            # Tier-1 poisoning
            # -------------------------------------------------

            if client_id in poisoned_clients:

                v, i, t = clients_raw_global[
                    client_id
                ]

                gate = data[
                    "gate"
                ]

                train_start, train_end = (
                    gate.train_idx
                )

                (
                    v_train_raw,
                    i_train_raw,
                    t_train_raw,
                ) = aligned_raw_segment(
                    v,
                    i,
                    t,
                    train_start,
                    train_end,
                )

                attack_rng = np.random.default_rng(
                    seed
                    + 200000
                    + round_idx * 1000
                    + client_id
                )

                (
                    v_atk,
                    i_atk,
                    t_atk,
                    _,
                ) = apply_tier1_attack(
                    v_train_raw,
                    i_train_raw,
                    t_train_raw,
                    "bias_injection",
                    attack_rng,
                    0,
                    len(
                        v_train_raw
                    ),
                )

                attack_gate = CrossModalGate(
                    ablation="full_cross_modal",
                    feature_mean=data[
                        "gate"
                    ].feature_mean,
                    feature_std=data[
                        "gate"
                    ].feature_std,
                    target_mean=data[
                        "gate"
                    ].target_mean,
                    target_std=data[
                        "gate"
                    ].target_std,
                )

                X_poison, y_poison = (
                    attack_gate._build_features(
                        v_atk,
                        i_atk,
                        t_atk,
                    )
                )

                X_train, y_train = (
                    standardize_arrays(
                        X_poison,
                        y_poison,
                        scaler_global,
                    )
                )

                if defended:

                    (
                        _,
                        _,
                        _,
                        keep_mask,
                    ) = data[
                        "gate"
                    ].verify_arrays(
                        X_poison,
                        y_poison,
                    )

            # -------------------------------------------------
            # Local training
            # -------------------------------------------------

            delta = train_local_model(
                global_state,
                X_train,
                y_train,
                keep_mask,
                config.LOCAL_EPOCHS,
                config.LEARNING_RATE,
                seed
                + client_id
                + round_idx * 10000,
                "full_cross_modal",
                return_delta=True,
            )

            # -------------------------------------------------
            # Tier-2 Byzantine attack
            # -------------------------------------------------

            if (
                tier2_attack is not None
                and client_id < f
            ):

                keys = sorted(
                    delta.keys()
                )

                flat_delta = flatten_state_dict(
                    delta,
                    keys,
                )

                attack_rng = np.random.default_rng(
                    seed
                    + 300000
                    + round_idx * 10000
                    + client_id
                )

                attacked_vector = (
                    apply_tier2_byzantine(
                        flat_delta.detach()
                        .cpu()
                        .numpy(),
                        tier2_attack,
                        attack_rng,
                    )
                )

                attacked_vector = torch.tensor(
                    attacked_vector,
                    dtype=flat_delta.dtype,
                )

                delta = unflatten_vector(
                    attacked_vector,
                    delta,
                )

            round_deltas.append(
                delta
            )

            if defended and client_id in poisoned_clients:

                retained = int(
                    np.sum(
                        keep_mask
                    )
                )

                sample_counts.append(
                    max(
                        1,
                        retained,
                    )
                )

            else:

                sample_counts.append(
                    data[
                        "n_train"
                    ]
                )

        # -----------------------------------------------------
        # Server aggregation
        # -----------------------------------------------------

        aggregated_delta = (
            aggregate_updates(
                round_deltas,
                aggregator,
                f,
                sample_counts=sample_counts,
            )
        )

        global_state = apply_delta(
            global_state,
            aggregated_delta,
        )

        # -----------------------------------------------------
        # Numerical sanity
        # -----------------------------------------------------

        for key, tensor in global_state.items():

            if not torch.isfinite(
                tensor
            ).all():

                raise FloatingPointError(
                    f"Non-finite global parameter after "
                    f"round {round_idx + 1}: {key}"
                )

    return global_state


# =========================================================
# GLOBAL VARIABLES USED BY THE FL ROUTINE
# =========================================================

clients_raw_global = None
scaler_global = None


# =========================================================
# E1
# =========================================================

def run_e1(
    gates,
):
    values = []

    for gate in gates:

        (
            _,
            _,
            anomaly_mask,
            _,
        ) = gate.verify(
            "val"
        )

        values.append(
            float(
                np.mean(
                    anomaly_mask
                )
            )
        )

    return values


# =========================================================
# ATTACK EVALUATION
# =========================================================

def evaluate_tier1_attack(
    gate,
    v,
    i,
    t,
    attack_type,
    seed_offset,
):
    test_start, test_end = (
        gate.test_idx
    )

    (
        v_test,
        i_test,
        t_test,
    ) = aligned_raw_segment(
        v,
        i,
        t,
        test_start,
        test_end,
    )

    attack_start = int(
        len(v_test)
        * config.ATTACK_FRACTION
    )

    attack_rng = np.random.default_rng(
        seed_offset
    )

    (
        v_atk,
        i_atk,
        t_atk,
        y_true_raw,
    ) = apply_tier1_attack(
        v_test,
        i_test,
        t_test,
        attack_type,
        attack_rng,
        attack_start,
        len(v_test),
    )

    X_attack, y_attack = (
        gate._build_features(
            v_atk,
            i_atk,
            t_atk,
        )
    )

    (
        _,
        _,
        anomaly_mask,
        _,
    ) = gate.verify_arrays(
        X_attack,
        y_attack,
    )

    y_true_aligned = (
        y_true_raw[1:]
    )

    return compute_classification_metrics(
        anomaly_mask,
        y_true_aligned,
    )


# =========================================================
# E2
# =========================================================

def run_e2(
    clients_raw,
    gates,
    seed,
):
    results = []

    for client_id in range(
        config.NUM_CLIENTS
    ):

        v, i, t = clients_raw[
            client_id
        ]

        gate = gates[
            client_id
        ]

        for attack_index, attack_type in enumerate(
            config.TIER1_ATTACKS
        ):

            metrics = evaluate_tier1_attack(
                gate,
                v,
                i,
                t,
                attack_type,
                seed
                + 100000
                + client_id * 100
                + attack_index,
            )

            results.append(
                {
                    "seed": seed,
                    "client": client_id,
                    "attack": attack_type,
                    "f1": metrics[
                        "f1"
                    ],
                    "precision": metrics[
                        "precision"
                    ],
                    "recall": metrics[
                        "recall"
                    ],
                    "fpr": metrics[
                        "fpr"
                    ],
                }
            )

    return results


# =========================================================
# E3
# =========================================================

def run_e3(
    clients_raw,
    gates,
    client_data,
    seed,
):
    global clients_raw_global
    global scaler_global

    clients_raw_global = clients_raw

    # Reconstruct the scaler from the gates.
    scaler_global = {
        "feature_mean": gates[0].feature_mean,
        "feature_std": gates[0].feature_std,
        "target_mean": gates[0].target_mean,
        "target_std": gates[0].target_std,
    }

    poisoned_clients = set(
        range(
            config.NUM_CLIENTS // 2
        )
    )

    clean_state = run_federated_training(
        client_data,
        seed,
        poisoned_clients=set(),
        defended=False,
        aggregator="fedavg",
    )

    undefended_state = run_federated_training(
        client_data,
        seed,
        poisoned_clients=poisoned_clients,
        defended=False,
        aggregator="fedavg",
    )

    defended_state = run_federated_training(
        client_data,
        seed,
        poisoned_clients=poisoned_clients,
        defended=True,
        aggregator="fedavg",
    )

    X_test = client_data[0][
        "X_test"
    ]

    y_test = client_data[0][
        "y_test"
    ]

    clean_mse = (
        evaluate_global_model_physical_mse(
            clean_state,
            X_test,
            y_test,
            "full_cross_modal",
            scaler_global[
                "target_std"
            ],
        )
    )

    undefended_mse = (
        evaluate_global_model_physical_mse(
            undefended_state,
            X_test,
            y_test,
            "full_cross_modal",
            scaler_global[
                "target_std"
            ],
        )
    )

    defended_mse = (
        evaluate_global_model_physical_mse(
            defended_state,
            X_test,
            y_test,
            "full_cross_modal",
            scaler_global[
                "target_std"
            ],
        )
    )

    return {
        "seed": seed,
        "clean_loss": clean_mse,
        "undefended_loss": undefended_mse,
        "defended_loss": defended_mse,
    }


# =========================================================
# E4
# =========================================================

def run_e4(
    clients_raw,
    gates,
    client_data,
    seed,
):
    global clients_raw_global
    global scaler_global

    clients_raw_global = clients_raw

    scaler_global = {
        "feature_mean": gates[0].feature_mean,
        "feature_std": gates[0].feature_std,
        "target_mean": gates[0].target_mean,
        "target_std": gates[0].target_std,
    }

    aggregators = [
        "fedavg",
        "coordinate_median",
        "trimmed_mean",
        "krum",
    ]

    results = []

    X_test = client_data[0][
        "X_test"
    ]

    y_test = client_data[0][
        "y_test"
    ]

    # -----------------------------------------------------
    # Clean baseline
    # -----------------------------------------------------

    clean_state = run_federated_training(
        client_data,
        seed,
        poisoned_clients=set(),
        defended=False,
        aggregator="fedavg",
    )

    clean_mse = (
        evaluate_global_model_physical_mse(
            clean_state,
            X_test,
            y_test,
            "full_cross_modal",
            scaler_global[
                "target_std"
            ],
        )
    )

    for attack_index, attack_type in enumerate(
        config.TIER2_ATTACKS
    ):

        for aggregator_index, aggregator in enumerate(
            aggregators
        ):

            state = run_federated_training(
                client_data,
                seed
                + attack_index * 100
                + aggregator_index,
                poisoned_clients=set(),
                defended=False,
                aggregator=aggregator,
                tier2_attack=attack_type,
            )

            mse = (
                evaluate_global_model_physical_mse(
                    state,
                    X_test,
                    y_test,
                    "full_cross_modal",
                    scaler_global[
                        "target_std"
                    ],
                )
            )

            results.append(
                {
                    "seed": seed,
                    "attack": attack_type,
                    "agg": aggregator,
                    "loss": mse,
                    "clean_baseline_loss": clean_mse,
                    "relative_degradation": (
                        mse / max(
                            clean_mse,
                            1e-12,
                        )
                    ),
                }
            )

    return results


# =========================================================
# E5
# =========================================================

def run_e5(
    seed,
):
    results = []

    for heterogeneity_scale in [
        0.5,
        1.0,
        2.0,
    ]:

        f1_scores = []

        for client_id in range(
            config.NUM_CLIENTS
        ):

            sim = MotorPhysicsSimulator(
                seed=seed + client_id,
                heterogeneity_scale=heterogeneity_scale,
            )

            v, i, t = (
                sim.generate_telemetry(
                    config.NUM_SAMPLES
                )
            )

            # Fit local scaler only from this client's training
            # partition. This is appropriate for E5 because the
            # experiment measures physical heterogeneity rather
            # than FL aggregation.
            scaler = fit_global_standardizer(
                [
                    (
                        v,
                        i,
                        t,
                    )
                ]
            )

            gate = fit_gate(
                v,
                i,
                t,
                "full_cross_modal",
                scaler,
            )

            metrics = evaluate_tier1_attack(
                gate,
                v,
                i,
                t,
                "bias_injection",
                seed
                + 400000
                + client_id
                + int(
                    heterogeneity_scale
                    * 100
                ),
            )

            f1_scores.append(
                metrics["f1"]
            )

        results.append(
            {
                "seed": seed,
                "heterogeneity": heterogeneity_scale,
                "mean_f1": float(
                    np.mean(
                        f1_scores
                    )
                ),
                "worst_f1": float(
                    np.min(
                        f1_scores
                    )
                ),
            }
        )

    return results


# =========================================================
# E6
# =========================================================

def run_e6(
    clients_raw,
    seed,
):
    results = []

    v, i, t = clients_raw[0]

    for ablation_index, ablation in enumerate(
        config.ABLATIONS
    ):

        scaler = fit_global_standardizer(
            clients_raw,
            ablation,
        )

        gate = fit_gate(
            v,
            i,
            t,
            ablation,
            scaler,
        )

        for attack_index, attack_type in enumerate(
            config.E6_ATTACKS
        ):

            metrics = evaluate_tier1_attack(
                gate,
                v,
                i,
                t,
                attack_type,
                seed
                + 500000
                + ablation_index * 100
                + attack_index,
            )

            results.append(
                {
                    "seed": seed,
                    "ablation": ablation,
                    "attack": attack_type,
                    "f1": metrics[
                        "f1"
                    ],
                    "precision": metrics[
                        "precision"
                    ],
                    "recall": metrics[
                        "recall"
                    ],
                    "fpr": metrics[
                        "fpr"
                    ],
                }
            )

    return results


# =========================================================
# E7
# =========================================================

def run_e7(
    gates,
    client_data,
    seed,
):
    gate = gates[0]

    X_train = client_data[0][
        "X_train"
    ]

    y_train = client_data[0][
        "y_train"
    ]

    result = profile_execution(
        gate,
        X_train,
        y_train,
        config.LOCAL_EPOCHS,
        config.LEARNING_RATE,
        seed,
        "full_cross_modal",
    )

    result["seed"] = seed

    return result


# =========================================================
# MAIN
# =========================================================

def run_all():

    print(
        "=" * 70
    )

    print(
        "FINITE-SAMPLE CONFORMAL CROSS-MODAL VERIFICATION"
    )

    print(
        "RESEARCH PIPELINE v2.0"
    )

    print(
        "=" * 70
    )

    pre_execution_audit()

    os.makedirs(
        config.RESULTS_DIR,
        exist_ok=True,
    )

    os.makedirs(
        config.PLOTS_DIR,
        exist_ok=True,
    )

    results_store = {
        "e1": [],
        "e2": [],
        "e3": [],
        "e4": [],
        "e5": [],
        "e6": [],
        "e7": [],
    }

    total_start = time.perf_counter()

    for seed in config.SEEDS:

        print(
            f"\n--- Seed {seed} ---"
        )

        seed_everything(
            seed
        )

        # -----------------------------------------------------
        # Generate client telemetry.
        # -----------------------------------------------------

        clients_raw = []

        for client_id in range(
            config.NUM_CLIENTS
        ):

            sim = MotorPhysicsSimulator(
                seed=seed + client_id,
                heterogeneity_scale=1.0,
            )

            v, i, t = (
                sim.generate_telemetry(
                    config.NUM_SAMPLES
                )
            )

            clients_raw.append(
                (
                    v,
                    i,
                    t,
                )
            )

        # -----------------------------------------------------
        # Shared training-only scaler.
        # -----------------------------------------------------

        scaler = fit_global_standardizer(
            clients_raw,
            "full_cross_modal",
        )

        # -----------------------------------------------------
        # Gates.
        # -----------------------------------------------------

        gates = []

        for client_id in range(
            config.NUM_CLIENTS
        ):

            v, i, t = clients_raw[
                client_id
            ]

            gate = fit_gate(
                v,
                i,
                t,
                "full_cross_modal",
                scaler,
            )

            gates.append(
                gate
            )

        # -----------------------------------------------------
        # Client FL data.
        # -----------------------------------------------------

        client_data = prepare_client_data(
            clients_raw,
            scaler,
            "full_cross_modal",
        )

        # -----------------------------------------------------
        # E1
        # -----------------------------------------------------

        print(
            "  E1..."
        )

        e1_values = run_e1(
            gates
        )

        for client_id, value in enumerate(
            e1_values
        ):

            results_store[
                "e1"
            ].append(
                {
                    "seed": seed,
                    "client": client_id,
                    "fpr": value,
                }
            )

        # -----------------------------------------------------
        # E2
        # -----------------------------------------------------

        print(
            "  E2..."
        )

        results_store[
            "e2"
        ].extend(
            run_e2(
                clients_raw,
                gates,
                seed,
            )
        )

        # -----------------------------------------------------
        # E3
        # -----------------------------------------------------

        print(
            "  E3..."
        )

        e3_result = run_e3(
            clients_raw,
            gates,
            client_data,
            seed,
        )

        results_store[
            "e3"
        ].append(
            e3_result
        )

        # -----------------------------------------------------
        # E4
        # -----------------------------------------------------

        print(
            "  E4..."
        )

        results_store[
            "e4"
        ].extend(
            run_e4(
                clients_raw,
                gates,
                client_data,
                seed,
            )
        )

        # -----------------------------------------------------
        # E5
        # -----------------------------------------------------

        print(
            "  E5..."
        )

        results_store[
            "e5"
        ].extend(
            run_e5(
                seed
            )
        )

        # -----------------------------------------------------
        # E6
        # -----------------------------------------------------

        print(
            "  E6..."
        )

        results_store[
            "e6"
        ].extend(
            run_e6(
                clients_raw,
                seed,
            )
        )

        # -----------------------------------------------------
        # E7
        # -----------------------------------------------------

        print(
            "  E7..."
        )

        results_store[
            "e7"
        ].append(
            run_e7(
                gates,
                client_data,
                seed,
            )
        )

        # -----------------------------------------------------
        # Progress checkpoint.
        # -----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - total_start
        )

        print(
            f"  [DONE] Seed {seed} "
            f"elapsed={elapsed:.2f}s"
        )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    results_path = (
        config.RESULTS_DIR
        / "raw_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            results_store,
            handle,
            indent=4,
        )

    manifest = {
        "framework_name": config.FRAMEWORK_NAME,
        "version": config.VERSION,
        "seeds": config.SEEDS,
        "n_seeds": len(
            config.SEEDS
        ),
        "raw_samples": config.NUM_SAMPLES,
        "aligned_samples": (
            config.NUM_SAMPLES - 1
        ),
        "num_clients": config.NUM_CLIENTS,
        "num_rounds": config.NUM_ROUNDS,
        "local_epochs": config.LOCAL_EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "alpha": config.ALPHA,
        "byzantine_f": config.BYZANTINE_F,
        "tier1_attacks": config.TIER1_ATTACKS,
        "tier2_attacks": config.TIER2_ATTACKS,
        "ablations": config.ABLATIONS,
        "normalization": (
            "global training-only standardization"
        ),
        "fl_update_type": "parameter_delta",
        "krum_requirement": (
            "N >= 2f + 3"
        ),
        "temporal_partition": True,
        "conformal_exchangeability_caveat": True,
    }

    manifest_path = (
        config.RESULTS_DIR
        / "experiment_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=4,
        )

    print(
        "\nExperiments completed successfully."
    )

    print(
        f"Raw results saved to: {results_path}"
    )

    print(
        f"Manifest saved to: {manifest_path}"
    )


if __name__ == "__main__":
    run_all()