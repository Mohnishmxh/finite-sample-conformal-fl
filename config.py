"""
config.py
---------
Central configuration for the finite-sample conformal
cross-modal verification + federated learning project.

This configuration is intentionally compatible with the
existing physics_simulator.py and artifact-generation code.
"""

from pathlib import Path


# =========================================================
# PROJECT METADATA
# =========================================================

FRAMEWORK_NAME = "Finite-Sample Conformal Cross-Modal Verification"
PROJECT_NAME = FRAMEWORK_NAME
VERSION = "2.0"


# =========================================================
# REPRODUCIBILITY
# =========================================================

SEEDS = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909
]

DETERMINISTIC_TORCH = True


# =========================================================
# DATASET
# =========================================================

NUM_SAMPLES = 5000

TRAIN_RATIO = 0.40
CAL_RATIO = 0.20
VAL_RATIO = 0.20
TEST_RATIO = 0.20


# =========================================================
# PHYSICAL SIMULATOR
# =========================================================

# Names required by the current physics_simulator.py.
BASE_R_TH = 0.50
BASE_C_TH = 100.0
BASE_R_W = 0.50

T_AMBIENT = 25.0
DT = 1.0

NOISE_V_STD = 0.5
NOISE_I_STD = 0.1
NOISE_T_STD = 0.05

# Descriptive aliases.
BASE_VOLTAGE = 220.0
BASE_CURRENT = 5.0
BASE_TEMPERATURE = 25.0

VOLTAGE_NOISE_STD = NOISE_V_STD
CURRENT_NOISE_STD = NOISE_I_STD
TEMPERATURE_NOISE_STD = NOISE_T_STD

AMBIENT_TEMPERATURE = T_AMBIENT
THERMAL_TIME_CONSTANT = BASE_R_TH * BASE_C_TH
POWER_SCALE = 1.0

HETEROGENEITY_R_SCALE = 0.20
HETEROGENEITY_C_SCALE = 0.20

VOLTAGE_DRIFT_STD = 0.01
CURRENT_DRIFT_STD = 0.01
TEMPERATURE_DRIFT_STD = 0.01


# =========================================================
# FEDERATED LEARNING
# =========================================================

NUM_CLIENTS = 10
NUM_ROUNDS = 20

LOCAL_EPOCHS = 3
BATCH_SIZE = 32

# Because the FL data are standardized, this learning rate
# is deliberately smaller than the previous raw-scale setup.
LEARNING_RATE = 0.01

GRADIENT_CLIP_NORM = 5.0


# =========================================================
# FEDERATED CLIENT WEIGHTING
# =========================================================

# FedAvg weights clients according to their number of
# retained training samples.
USE_SAMPLE_WEIGHTED_FEDAVG = True


# =========================================================
# BYZANTINE THREAT MODEL
# =========================================================

BYZANTINE_FRACTION = 0.20
BYZANTINE_F = int(NUM_CLIENTS * BYZANTINE_FRACTION)

# Classical Krum requires N >= 2f + 3.
assert NUM_CLIENTS >= 2 * BYZANTINE_F + 3


# =========================================================
# CONFORMAL VERIFICATION
# =========================================================

ALPHA = 0.05

RIDGE_ALPHA = 1.0


# =========================================================
# NORMALIZATION
# =========================================================

NORMALIZATION_EPS = 1e-8


# =========================================================
# TIER-1 PHYSICAL ATTACKS
# =========================================================

GAUSSIAN_NOISE_STD = 2.0
BIAS_MAGNITUDE = 5.0
DESYNC_SHIFT = 5

TIER1_ATTACKS = [
    "gaussian_noise",
    "bias_injection",
    "desynchronize_current",
]


# =========================================================
# TIER-2 BYZANTINE ATTACKS
# =========================================================

TIER2_SIGN_FLIP = -1.0
TIER2_SCALING = 10.0
TIER2_ADDITIVE_NOISE_STD = 0.10

TIER2_ATTACKS = [
    "sign_flip",
    "scaling",
    "additive_noise",
    "zero",
]


# =========================================================
# ABLATIONS
# =========================================================

ABLATIONS = [
    "full_cross_modal",
    "missing_current",
    "missing_voltage",
    "no_temporal_lag",
    "single_modal_ar",
]


# =========================================================
# EVALUATION SETTINGS
# =========================================================

# E2/E5/E6 attacks begin halfway through the held-out
# test interval.
ATTACK_FRACTION = 0.50

# E6 evaluates every ablation against every Tier-1 attack.
E6_ATTACKS = list(TIER1_ATTACKS)


# =========================================================
# OUTPUT PATHS
# =========================================================

OUTPUT_DIR = Path("artifacts")

RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Compatibility with the previously generated directory.
LEGACY_RESULTS_DIR = Path("results")


# =========================================================
# SANITY CHECKS
# =========================================================

_PARTITION_SUM = (
    TRAIN_RATIO
    + CAL_RATIO
    + VAL_RATIO
    + TEST_RATIO
)

assert abs(_PARTITION_SUM - 1.0) < 1e-9

assert NUM_SAMPLES >= 2

assert NUM_CLIENTS >= 2 * BYZANTINE_F + 3

assert 0.0 < ALPHA < 1.0
