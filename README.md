Finite-Sample Conformal Cross-Modal Verification for Robust Cyber-Physical Federated Learning: Research Artifacts

This repository hosts the official experimental artifacts, aggregated results, and 300 DPI publication plots for the paper "Finite-Sample Conformal Cross-Modal Verification for Robust Cyber-Physical Federated Learning"[cite: 4].

Repository Structure

├── config.py                # Hyperparameters, paths, random seeds, and attack configs
├── physics_simulator.py     # Continuous motor telemetry & LPTN ODE simulator
├── crossmodal_gate.py       # Conformal predictor & thermodynamic residual gate
├── fl_engine.py             # FL local training, standardizers, and aggregation rules
├── attacks.py               # Tier-1 sensor spoofing & Tier-2 Byzantine attack generators
├── profiler.py              # Execution latency and peak heap memory profiler
├── run_experiments.py       # Multi-seed experimental runner (Tracks E1–E7)
├── generate_artifacts.py    # Publication-ready figure & CSV generation script
├── check_results.py         # Summary validator and terminal reporter
│── plots/               # Generated 300 DPI publication plots (Figures 1–7)
│── results/             # Raw JSON checkpoints and summary_results.csv
└── README.md

------------------------------------------------------------------------------------------------------------------------
Experimental Tracks (E1–E7)

The experimental testbed evaluates 7 complementary tracks across 10 random seeds and 100 client evaluations[cite: 4]:

E1 (Conformal FPR Control): Validates empirical false-positive rate control against nominal $\alpha = 0.05$ under clean physical trajectories[cite: 4].

E2 (Physical Anomaly Verification): Evaluates anomaly classification F1 score across sensor bias ($+5^\circ\text{C}$), Gaussian noise ($3\sigma$), and current desynchronization ($\Delta t = 5$)[cite: 4].

E3 (Federated Model Robustness): Measures global physical MSE under 50% compromised client nodes, identifying the sample-starvation penalty of hard pruning[cite: 4].

E4 (Byzantine Server Aggregation): Evaluates a 16-configuration matrix combining 4 aggregators (FedAvg, Coordinate Median, Trimmed Mean, Krum) against 4 Byzantine attack families[cite: 4].

E5 (Motor Parameter Heterogeneity): Assesses gate stability across motor scaling multipliers ($0.5\times$, $1.0\times$, $2.0\times$)[cite: 4].

E6 (Feature Ablation Study): Evaluates macro F1 after removing voltage, temporal difference lags, current, and cross-modal signals[cite: 4].

E7 (Software Overhead Profiling): Measures on-device verification execution latency (ms) and peak heap memory footprint (KB) against local SGD training[cite: 4].

----------------------------------------------------------------------------------------------------------------------------------------------------
Validated 10-Seed Summary Results

The table below summarizes the multi-seed experimental outcomes ($N=10$ seeds, $N=100$ client runs)[cite: 4]:

Track|Experiment/Setting|Metric|Mean|Std (±1σ)|95% CI
E1,Clean Fleet FPR (Fleet Means)[cite: 4],FPR,0.0642[cite: 4],0.0277[cite: 4],±0.0172[cite: 4]
E1,Clean Fleet FPR (Pooled Runs)[cite: 4],FPR,0.0642[cite: 4],0.0917[cite: 4],±0.0180[cite: 4]
E2,Thermal Bias Injection (+5∘C)[cite: 4],F1,0.9730[cite: 4],0.0044[cite: 4],±0.0027[cite: 4]
E2,Current Desynchronization (Δt=5)[cite: 4],F1,0.6679[cite: 4],0.0912[cite: 4],±0.0565[cite: 4]
E2,Gaussian Sensor Noise (3σ)[cite: 4],F1,0.9712[cite: 4],0.0050[cite: 4],±0.0031[cite: 4]
E3,Clean Baseline Reference[cite: 4],Test MSE (∘C2)[cite: 4],43.65[cite: 4],27.04[cite: 4],±16.76[cite: 4]
E3,Undefended FedAvg (50% Poisoned)[cite: 4],Test MSE (∘C2)[cite: 4],43.35[cite: 4],29.31[cite: 4],±18.17[cite: 4]
E3,Defended FedAvg (Sample Starvation)[cite: 4],Test MSE (∘C2)[cite: 4],50.44[cite: 4],32.99[cite: 4],±20.45[cite: 4]
E4,Additive Noise → Trimmed Mean[cite: 4],Test MSE (∘C2)[cite: 4],39.46[cite: 4],22.70[cite: 4],±14.07[cite: 4]
E4,Scaling (γ=10) → Trimmed Mean[cite: 4],Test MSE (∘C2)[cite: 4],32.01[cite: 4],10.55[cite: 4],±6.54[cite: 4]
E4,Sign Inversion → Coordinate Median[cite: 4],Test MSE (∘C2)[cite: 4],48.97[cite: 4],28.29[cite: 4],±17.54[cite: 4]
E4,Zero Update → Krum (Collapse)[cite: 4],Test MSE (∘C2)[cite: 4],1376.61[cite: 4],3847.63[cite: 4],±2384.79[cite: 4]
E5,Heterogeneity (2.0× Physical Scale)[cite: 4],Worst F1[cite: 4],0.9076[cite: 4],0.0639[cite: 4],±0.0396[cite: 4]
E6,Missing Current Telemetry (I)[cite: 4],Macro F1[cite: 4],0.1766[cite: 4],0.1724[cite: 4],±0.1068[cite: 4]
E6,Full Cross-Modal Gate[cite: 4],Macro F1[cite: 4],0.9031[cite: 4],0.1137[cite: 4],±0.0705[cite: 4]
E7,Gate Verification Latency[cite: 4],Latency (ms)[cite: 4],0.77[cite: 4],0.52[cite: 4],±0.32[cite: 4]
E7,Local FL Training (3 Epochs)[cite: 4],Latency (ms)[cite: 4],310.01[cite: 4],88.59[cite: 4],±54.91[cite: 4]
E7,Gate Peak Heap Memory[cite: 4],Heap (KB)[cite: 4],439.5[cite: 4],0.02[cite: 4],±0.01[cite: 4]
E7,Training Peak Heap Memory[cite: 4],Heap (KB)[cite: 4],168.9[cite: 4],2.17[cite: 4],±1.34[cite: 4]

---------------------------------------------------------------------------------------------------------------------------

Artifact Descriptions

fig1_e1_clean_fpr.png (Figure 1): Fleet empirical false-positive rate across 10 client nodes[cite: 4]. Validates alignment with nominal $\alpha = 0.05$ ($0.064$ fleet mean) while documenting tail spread up to $0.137$ during non-stationary regime transitions[cite: 4].

fig2_e2_detection_performance.png (Figure 2): Sensor verification F1 scores[cite: 4]. Demonstrates high detection power on magnitude anomalies (bias $0.973$, noise $0.971$) alongside the steady-state current plateau blind spot ($0.668$)[cite: 4].

fig3_e3_fl_robustness.png (Figure 3): Global model predictive MSE under 50% Tier-1 poisoning[cite: 4]. Quantifies the sample-starvation penalty: filtering 100% of poisoned clients inflates participation variance, increasing MSE from $43.35^\circ\text{C}^2$ to $50.44^\circ\text{C}^2$[cite: 4].

fig4_e4_aggregation_robustness.png (Figure 4): Evaluation across 16 Byzantine attack-defense configurations[cite: 4]. Truncated at $270^\circ\text{C}^2$ to contrast Trimmed Mean stability ($32.0\text{--}58.9^\circ\text{C}^2$) against Krum's catastrophic collapse under zero-update attacks ($1376.6^\circ\text{C}^2$)[cite: 4].

fig5_e5_heterogeneity.png (Figure 5): Gate robustness across hardware variations ($0.5\times$ to $2.0\times$ thermal resistance/capacitance)[cite: 4]. Shows fleet mean holding above $0.969$ and worst-client F1 remaining above $0.90$[cite: 4].

fig6_e6_ablation_study.png (Figure 6): Feature ablation analysis confirming that Joule heating ($I^2 R_w$) is the indispensable grounding signal[cite: 4]. Removing voltage or temporal differences incurs $<0.15\%$ change, whereas excluding current drops macro F1 to $0.177$[cite: 4].

fig7_e7_overhead.png (Figure 7): Microcontroller overhead profile[cite: 4]. Highlights that execution latency is negligible ($0.77\text{ ms}$, $0.26\%$ of training time), while sliding-window buffering introduces a $2.6\times$ peak heap RAM footprint[cite: 4].

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Data & Simulation Provenance

All telemetry and physical trajectories were generated via numerical simulation of first-order lumped-parameter thermal network dynamics across independent multi-seed runs[cite: 4]. No external proprietary datasets were used, no human subjects were involved, and no personal data was collected[cite: 4]. High-frequency iron losses and spatial rotor gradients are abstracted into effective parameters ($R_{th}, C_{th}$) to isolate thermodynamic consistency checking[cite: 4].


