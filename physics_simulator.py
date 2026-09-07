"""
physics_simulator.py
--------------------
Cyber-physical motor simulator modeling thermal inertia, heterogeneous
physical parameters, and measurement noise.

The simulator produces finite deterministic telemetry for a given seed.
"""

import numpy as np
import config


class MotorPhysicsSimulator:
    """
    Simulates a simplified motor thermal system.

    State:
        T_t = winding / motor temperature at time t

    Electrical inputs:
        V_t = voltage
        I_t = current

    Thermal dynamics:
        C_th * dT/dt =
            P_loss - (T - T_ambient) / R_th

    where:

        P_loss = I^2 * R_w
    """

    def __init__(self, seed: int, heterogeneity_scale: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.scale = heterogeneity_scale

        # Heterogeneous but physically bounded parameters.
        r_factor_low = 1.0 - 0.2 * self.scale
        r_factor_high = 1.0 + 0.2 * self.scale

        self.r_th = max(
            0.1,
            config.BASE_R_TH
            * self.rng.uniform(r_factor_low, r_factor_high)
        )

        self.c_th = max(
            1.0,
            config.BASE_C_TH
            * self.rng.uniform(r_factor_low, r_factor_high)
        )

        self.r_w = max(
            0.05,
            config.BASE_R_W
            * self.rng.uniform(r_factor_low, r_factor_high)
        )

        self.t_amb = (
            config.T_AMBIENT
            + self.rng.normal(0.0, 2.0 * self.scale)
        )

    def generate_telemetry(self, num_samples: int):
        """
        Generate voltage, current, and temperature telemetry.

        Returns
        -------
        v : np.ndarray
            Voltage telemetry.
        i : np.ndarray
            Current telemetry.
        t : np.ndarray
            Temperature telemetry.
        """

        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")

        v = np.zeros(num_samples, dtype=float)
        i = np.zeros(num_samples, dtype=float)
        t = np.zeros(num_samples, dtype=float)

        t[0] = self.t_amb

        regime = 0.0

        for step in range(num_samples):

            # Change operating regime periodically.
            if step % 500 == 0:
                regime = float(
                    self.rng.choice(
                        [1.0, 1.5, 0.5]
                    )
                )

            # Nominal electrical state.
            v_true = 220.0 * regime

            # Random effective impedance.
            effective_resistance = self.rng.uniform(10.0, 15.0)

            # Nonlinear current proxy.
            i_true = (
                v_true / max(0.1, effective_resistance)
            ) * regime

            # Noisy measurements.
            v[step] = (
                v_true
                + self.rng.normal(0.0, config.NOISE_V_STD)
            )

            i[step] = (
                i_true
                + self.rng.normal(0.0, config.NOISE_I_STD)
            )

            # Thermal state update.
            if step > 0:

                p_loss = (
                    i[step - 1] ** 2
                ) * self.r_w

                heat_dissipation = (
                    (t[step - 1] - self.t_amb)
                    / self.r_th
                )

                t_dot = (
                    p_loss - heat_dissipation
                ) / self.c_th

                t_next = (
                    t[step - 1]
                    + t_dot * config.DT
                    + self.rng.normal(
                        0.0,
                        config.NOISE_T_STD
                    )
                )

                # Physical / numerical bounds.
                t[step] = np.clip(
                    t_next,
                    self.t_amb - 10.0,
                    400.0
                )

        # Sanity checks.
        assert np.isfinite(v).all(), (
            "Voltage telemetry contains NaN/Inf."
        )

        assert np.isfinite(i).all(), (
            "Current telemetry contains NaN/Inf."
        )

        assert np.isfinite(t).all(), (
            "Temperature telemetry contains NaN/Inf."
        )

        assert np.var(t) > 1e-4, (
            "Temperature telemetry is numerically degenerate."
        )

        return v, i, t