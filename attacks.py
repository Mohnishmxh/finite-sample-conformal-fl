
"""
attacks.py
----------
Tier-1 physical telemetry attacks and Tier-2
Byzantine model-update attacks.
"""

import numpy as np


# =========================================================
# TIER-1 ATTACKS
# =========================================================

def apply_tier1_attack(
    v,
    i,
    t,
    attack_type,
    rng,
    attack_start,
    attack_end
):
    """
    Apply a physical telemetry attack only to the
    requested raw interval.

    Returns:

        v_attacked
        i_attacked
        t_attacked
        y_true_raw

    y_true_raw marks the raw samples that were directly
    modified by the attack.
    """

    v = np.asarray(
        v,
        dtype=np.float64
    ).copy()

    i = np.asarray(
        i,
        dtype=np.float64
    ).copy()

    t = np.asarray(
        t,
        dtype=np.float64
    ).copy()

    y_true_raw = np.zeros(
        len(v),
        dtype=bool
    )

    start = max(
        0,
        int(attack_start)
    )

    end = min(
        len(v),
        int(attack_end)
    )

    if start >= end:
        raise ValueError(
            "Invalid attack interval."
        )

    y_true_raw[
        start:end
    ] = True

    attack_type = (
        attack_type.lower()
    )

    # -----------------------------------------------------
    # Gaussian noise
    # -----------------------------------------------------

    if attack_type == "gaussian_noise":

        v[start:end] += rng.normal(
            0.0,
            2.0,
            size=end - start
        )

        i[start:end] += rng.normal(
            0.0,
            2.0,
            size=end - start
        )

        t[start:end] += rng.normal(
            0.0,
            2.0,
            size=end - start
        )

    # -----------------------------------------------------
    # Temperature bias
    # -----------------------------------------------------

    elif attack_type == "bias_injection":

        t[start:end] += 5.0

    # -----------------------------------------------------
    # Current desynchronization
    # -----------------------------------------------------

    elif attack_type == "desynchronize_current":

        shift = 5

        original = i.copy()

        attacked_segment = original[
            start:end
        ]

        if len(
            attacked_segment
        ) > shift:

            i[
                start + shift:end
            ] = attacked_segment[
                :len(attacked_segment) - shift
            ]

            i[
                start:start + shift
            ] = attacked_segment[
                0
            ]

        else:

            i[start:end] = (
                attacked_segment[0]
            )

    elif attack_type in {
        "none",
        "clean"
    }:

        pass

    else:

        raise ValueError(
            f"Unknown Tier-1 attack: "
            f"{attack_type}"
        )

    return (
        v,
        i,
        t,
        y_true_raw
    )


# =========================================================
# TIER-2 BYZANTINE UPDATE ATTACKS
# =========================================================

def apply_tier2_byzantine(
    vector,
    attack_type,
    rng
):
    """
    Apply an attack to the COMPLETE flattened
    client update vector.

    This is intentionally performed on the whole
    vector rather than separately on individual
    state tensors.
    """

    vector = np.asarray(
        vector,
        dtype=np.float64
    )

    attack_type = (
        attack_type.lower()
    )

    # -----------------------------------------------------
    # Sign flip
    # -----------------------------------------------------

    if attack_type == "sign_flip":

        return -vector

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    if attack_type == "scaling":

        return 10.0 * vector

    # -----------------------------------------------------
    # Additive noise
    # -----------------------------------------------------

    if attack_type == "additive_noise":

        noise = rng.normal(
            0.0,
            0.10,
            size=vector.shape
        )

        return vector + noise

    # -----------------------------------------------------
    # Zero update
    # -----------------------------------------------------

    if attack_type == "zero":

        return np.zeros_like(
            vector
        )

    raise ValueError(
        f"Unknown Tier-2 attack: "
        f"{attack_type}"
    )


# =========================================================
# ATTACK LISTS
# =========================================================

TIER1_ATTACKS = [
    "gaussian_noise",
    "bias_injection",
    "desynchronize_current",
]

TIER2_ATTACKS = [
    "sign_flip",
    "scaling",
    "additive_noise",
    "zero",
]

