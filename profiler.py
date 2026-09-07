"""
profiler.py
-----------
Software-level profiling of gate verification versus local FL training.

This measures Python process execution time and tracemalloc memory.
It does NOT establish bare-metal hardware latency, energy consumption,
sensor I/O latency, or real-time guarantees.
"""

import time
import tracemalloc

import fl_engine


def profile_execution(
    gate,
    X_train,
    y_train,
    epochs,
    lr,
    seed,
    ablation
):
    """
    Compare gate verification and local FL training overhead.
    """

    # =====================================================
    # GATE VERIFICATION
    # =====================================================

    tracemalloc.start()

    start_time = time.perf_counter()

    gate.verify_arrays(
        X_train,
        y_train
    )

    verify_time = (
        time.perf_counter()
        - start_time
    )

    _, peak_mem_verify = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    # =====================================================
    # LOCAL FL TRAINING
    # =====================================================

    model = fl_engine.FLModel(
        ablation
    )

    global_weights = model.state_dict()

    tracemalloc.start()

    start_time = time.perf_counter()

    fl_engine.train_local_model(
        global_weights,
        X_train,
        y_train,
        None,
        epochs,
        lr,
        seed,
        ablation
    )

    train_time = (
        time.perf_counter()
        - start_time
    )

    _, peak_mem_train = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    ratio = (
        verify_time
        / max(
            1e-9,
            train_time
        )
    )

    return {
        "verify_time_s": float(
            verify_time
        ),
        "train_time_s": float(
            train_time
        ),
        "verification_to_local_training_time_ratio": float(
            ratio
        ),
        "peak_mem_verify_bytes": int(
            peak_mem_verify
        ),
        "peak_mem_train_bytes": int(
            peak_mem_train
        )
    }