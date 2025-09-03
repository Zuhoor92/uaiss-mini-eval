#!/usr/bin/env python3
# UAISS Mini Evaluation — Offline, reproducible
# Generates synthetic UAV data, injects 3 attack windows, runs detectors mapped to UAISS L2/L3/L4,
# and produces figures + a metrics CSV for your paper.

import os, csv, math
import numpy as np

# Matplotlib: no seaborn, one plot per figure, no custom colors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- Config --------
T = 600                    # seconds of simulated flight
OUT = "assets"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# -------- 1) Synthetic data --------
t = np.arange(T)

# Smooth baseline motion
vx_base = 8 + 0.5 * np.sin(t/40)         # ~8 m/s forward
vy_base = 0.8 * np.cos(t/55)             # slight lateral
ax_noise = rng.normal(0, 0.15, size=T)   # accel noise
ay_noise = rng.normal(0, 0.12, size=T)

vx = vx_base + np.cumsum(ax_noise)*0.02
vy = vy_base + np.cumsum(ay_noise)*0.02
x = np.cumsum(vx)
y = np.cumsum(vy)

alt = 100 + np.cumsum(rng.normal(0, 0.05, size=T))  # ~100 m drift
yaw = np.cumsum(rng.normal(0, 0.05, size=T))        # small yaw drift

# Attack windows
gps_attack_start, gps_attack_end = 200, 260   # GPS spoofing
cmd_attack_start, cmd_attack_end = 320, 340   # Command injection
tel_attack_start, tel_attack_end = 420, 460   # Telemetry anomaly

# Inject GPS spoofing: position jumps
for k in range(gps_attack_start, gps_attack_end, 5):
    jump_mag = rng.uniform(80, 150)  # meters
    ang = rng.uniform(0, 2*np.pi)
    x[k:] += jump_mag * math.cos(ang)
    y[k:] += jump_mag * math.sin(ang)

# Inject telemetry anomalies: altitude/yaw spikes
alt[tel_attack_start:tel_attack_end] += np.sin(
    np.linspace(0, 15*np.pi, tel_attack_end - tel_attack_start)
) * 6
yaw[tel_attack_start:tel_attack_end] += rng.normal(
    0, 0.9, size=(tel_attack_end - tel_attack_start)
)

# Command stream (1 Hz)
allowed_cmds = ["ARM", "TAKEOFF", "NAV", "HOLD", "RTL", "LAND", "DISARM"]
attack_cmds = ["RAW_PWM", "OVERRIDE", "DIRECT_SETPOINT", "SPOOF_CMD"]
cmds = []
for i in range(T):
    if cmd_attack_start <= i < cmd_attack_end and rng.random() < 0.8:
        cmd = rng.choice(attack_cmds).item()
        session_ok = rng.random() > 0.3
        role = "attacker"
    else:
        cmd = rng.choice(allowed_cmds).item()
        session_ok = True
        role = "operator"
    cmds.append((i, cmd, "valid" if session_ok else "missing", role))

# Ground-truth labels
gt_gps = np.zeros(T, dtype=int); gt_gps[gps_attack_start:gps_attack_end] = 1
gt_cmd = np.zeros(T, dtype=int); gt_cmd[cmd_attack_start:cmd_attack_end] = 1
gt_tel = np.zeros(T, dtype=int); gt_tel[tel_attack_start:tel_attack_end] = 1

# Save raw CSV (good for Appendix)
with open("uav_timeseries.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_sec","x_m","y_m","alt_m","yaw_rad","cmd","session","role",
                "gt_gps_spoof","gt_cmd_inject","gt_tel_anom"])
    for i in range(T):
        w.writerow([i, float(x[i]), float(y[i]), float(alt[i]), float(yaw[i]),
                    cmds[i][1], cmds[i][2], cmds[i][3],
                    int(gt_gps[i]), int(gt_cmd[i]), int(gt_tel[i])])

# -------- 2) Detectors (UAISS L2/L3/L4) --------
# Helper kinematics
dx = np.diff(x, prepend=x[0])
dy = np.diff(y, prepend=y[0])
dist = np.sqrt(dx**2 + dy**2)  # m/s
heading = np.degrees(np.arctan2(dy, dx))
dhead = np.abs(np.diff(heading, prepend=heading[0]))
dhead = np.where(dhead > 180, 360 - dhead, dhead)

# L3 – Signal Integrity (GPS spoofing): thresholds
gps_pred = ((dist > 50) | (dhead > 100)).astype(int)

# L2 – RBAC/Session (Command injection): allowlist + token + role
cmd_pred = np.zeros(T, dtype=int)
for i, (_t, cmd, session, role) in enumerate(cmds):
    cmd_pred[i] = 1 if (cmd not in allowed_cmds or session != "valid" or role == "attacker") else 0

# L4 – AI/IDS (Telemetry anomalies): robust z-score on features
def robust_z(v):
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-6
    return np.abs((v - med) / (1.4826 * mad))

speed    = dist
alt_rate = np.diff(alt, prepend=alt[0])
yaw_rate = np.diff(yaw, prepend=yaw[0])

rz_speed = robust_z(speed)
rz_alt   = robust_z(alt_rate)
rz_yaw   = robust_z(yaw_rate)

tel_pred = ((rz_speed > 6) | (rz_alt > 6) | (rz_yaw > 6)).astype(int)

# -------- 3) Metrics --------
def metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp+fp) else 0.0
    rec  = tp / (tp + fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0
    return tp, fp, tn, fn, prec, rec, f1

m_gps = metrics(gt_gps, gps_pred)
m_cmd = metrics(gt_cmd, cmd_pred)
m_tel = metrics(gt_tel, tel_pred)

with open("results_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Attack Type","Layer/Detector","TP","FP","TN","FN","Precision","Recall","F1"])
    w.writerow(["GPS Spoofing","L3 – Signal Integrity", *m_gps])
    w.writerow(["Command Injection","L2 – RBAC/Session", *m_cmd])
    w.writerow(["Telemetry Anomaly","L4 – AI/Anomaly (Robust-Z)", *m_tel])

# -------- 4) Plots --------
# Grouped bar: Precision/Recall/F1 per attack
labels = ["GPS Spoof", "Cmd Inject", "Telemetry"]
precisions = [m_gps[4], m_cmd[4], m_tel[4]]
recalls    = [m_gps[5], m_cmd[5], m_tel[5]]
f1s        = [m_gps[6], m_cmd[6], m_tel[6]]

fig = plt.figure(figsize=(7,4))
xpos = np.arange(len(labels)); width = 0.6/3
plt.bar(xpos - width, precisions, width, label="Precision")
plt.bar(xpos,         recalls,    width, label="Recall")
plt.bar(xpos + width, f1s,        width, label="F1")
plt.xticks(xpos, labels)
plt.title("Detection Metrics by Attack Type")
plt.ylabel("Score (0–1)")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT, "detection_metrics.png"), dpi=200)
plt.close(fig)

# Timelines
def timeline(gt, pred, title, fname):
    fig = plt.figure(figsize=(8,3))
    plt.plot(t, gt, linestyle="-", label="Ground Truth")
    plt.plot(t, pred, linestyle="--", label="Detected")
    plt.xlabel("Time (s)"); plt.ylabel("Attack (0/1)")
    plt.title(title); plt.ylim(-0.1, 1.1); plt.legend(); plt.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200); plt.close(fig)

timeline(gt_gps, gps_pred, "GPS Spoofing – Ground Truth vs Detection", "timeline_gps.png")
timeline(gt_cmd, cmd_pred, "Command Injection – Ground Truth vs Detection", "timeline_cmd.png")
timeline(gt_tel, tel_pred, "Telemetry Anomaly – Ground Truth vs Detection", "timeline_tel.png")

# Layer–attack coverage matrix
coverage = np.array([
    [0,0,0],  # L1 AES (support)
    [0,1,0],  # L2 RBAC
    [1,0,0],  # L3 Signal
    [0,0,1],  # L4 AI/IDS
    [0,0,0],  # L5 Logging (support)
], dtype=float)

fig = plt.figure(figsize=(5,3.5))
plt.imshow(coverage, aspect="auto")
plt.xticks([0,1,2], labels)
plt.yticks([0,1,2,3,4], ["L1 AES","L2 RBAC","L3 Signal","L4 AI/IDS","L5 Logging"])
plt.title("Layer–Attack Coverage Matrix (1=Detects)")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "layer_coverage_matrix.png"), dpi=200)
plt.close(fig)

print("Done.\n- Figures: assets/\n- Metrics: results_summary.csv\n- Data: uav_timeseries.csv")
