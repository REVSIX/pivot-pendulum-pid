"""Dual-motor rack-and-pinion elevator: tune PID gains, then replay in Pivot.

Two ideal pinions each have radius 0.04 m. Each actuator command is motor
TORQUE (N*m), mapped to rack FORCE by MuJoCo's gear=1/r=25. The pinions are
visual representations, not tooth-contact or rotating-gear simulations.

Run: python train_elevator.py
Pivot embedded: train_elevator.py --pivot-live (uses existing Pivot env vars)
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

try:
    import ide_telem as ide
except ImportError:
    class _NoTelemetry:
        @staticmethod
        def log(*args, **kwargs):
            pass
    ide = _NoTelemetry()

EMBEDDED = "--pivot-live" in sys.argv
if EMBEDDED:
    sys.stdout = sys.stderr  # Reserve real stdout for Pivot frame protocol.

MODEL_PATH = Path(os.environ.get("PIVOT_MODEL_PATH", Path(__file__).with_name("elevator.xml")))
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
assert model.nq == model.nv == 1 and model.nu == 2, "Expected one slide and two actuators"
DT = model.opt.timestep
EPISODE_SECONDS = 8.0
EPISODE_STEPS = round(EPISODE_SECONDS / DT)
EPOCHS = 150
UPDATE_EVERY = 10
PINION_RADIUS = 0.04
MASS = float(np.sum(model.body_mass[1:]))
GRAVITY = abs(float(model.opt.gravity[2]))
# Setpoint sequence (seconds, meters). Repeat from the initial height each episode.
TRAJECTORY = [(0.0, 0.20), (1.0, 1.20), (3.5, 0.55), (5.5, 1.50)]
INITIAL_GAINS = np.array([35.0, 0.0, 9.0], dtype=float)  # Kp, Ki, Kd in force units


def target_at(t):
    target = TRAJECTORY[0][1]
    for change_time, value in TRAJECTORY:
        if t >= change_time:
            target = value
        else:
            break
    return target


def reset(data):
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.20
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)
    return 0.0  # controller integral


def control(data, gains, integral):
    x, velocity = float(data.qpos[0]), float(data.qvel[0])
    error = target_at(data.time) - x
    integral = float(np.clip(integral + error * DT, -1.5, 1.5))
    kp, ki, kd = gains
    # Gravity feedforward is required: otherwise an elevator settles below its setpoint.
    force = MASS * GRAVITY + kp * error + ki * integral - kd * velocity
    # Both pinions share effort equally: torque = force * radius / 2.
    per_motor_torque = 0.5 * force * PINION_RADIUS
    for motor in range(2):
        lo, hi = model.actuator_ctrlrange[motor]
        data.ctrl[motor] = np.clip(per_motor_torque, lo, hi)
    return integral, error, float(data.ctrl[0]), float(data.ctrl[1])


def step(data, gains, integral):
    integral, error, torque1, torque2 = control(data, gains, integral)
    mujoco.mj_step(model, data)
    new_error = target_at(data.time) - float(data.qpos[0])
    # Accuracy + settling / overshoot + motor effort.
    loss = (20.0 * new_error**2 + 0.4 * float(data.qvel[0])**2
            + 0.02 * (torque1**2 + torque2**2)) * DT
    return integral, loss, new_error


def rollout(gains):
    data = mujoco.MjData(model)
    integral = reset(data)
    loss = 0.0
    for _ in range(EPISODE_STEPS):
        integral, delta, _ = step(data, gains, integral)
        loss += delta
        if not np.isfinite(loss):
            return 1e9
    return loss


lock = threading.Lock()
start_training = threading.Event()
stop_training = threading.Event()
published_gains = INITIAL_GAINS.copy()
published_epoch = 0
latest_epoch = 0
finished = False


def training_worker():
    global published_gains, published_epoch, latest_epoch, finished
    start_training.wait()
    if stop_training.is_set():
        return
    params = torch.nn.Parameter(torch.tensor(INITIAL_GAINS, dtype=torch.float64))
    optimizer = torch.optim.Adam([params], lr=0.35)
    best_gains = INITIAL_GAINS.copy()
    best_loss = rollout(best_gains)
    print(f"Epoch 000 | Initial loss {best_loss:.4f}")
    try:
        for epoch in range(1, EPOCHS + 1):
            if stop_training.is_set():
                break
            current = params.detach().numpy().copy()
            loss = rollout(current)
            gradient = np.zeros(3)
            for j in range(3):
                perturb = max(0.02, abs(current[j]) * 0.005)
                plus, minus = current.copy(), current.copy()
                plus[j] += perturb
                minus[j] -= perturb
                gradient[j] = (rollout(plus) - rollout(minus)) / (2 * perturb)
            optimizer.zero_grad()
            params.grad = torch.tensor(gradient, dtype=torch.float64)
            optimizer.step()
            with torch.no_grad():
                params.clamp_(0.0, 250.0)
            if loss < best_loss:
                best_loss, best_gains = loss, current.copy()
            with lock:
                latest_epoch = epoch
                if epoch % UPDATE_EVERY == 0 or epoch == EPOCHS:
                    # Current gains (not just best) let us watch both progress and setbacks.
                    published_gains = params.detach().numpy().copy()
                    published_epoch = epoch
            ide.log("training_loss", float(loss))
            ide.log("best_training_loss", float(best_loss))
            ide.log("training_epoch", epoch)
            if epoch % UPDATE_EVERY == 0:
                print(f"Epoch {epoch:03d} | loss={loss:.4f} | gains={params.detach().tolist()}")
        print(f"Training done. Best gains: {best_gains.tolist()} | loss={best_loss:.4f}")
    finally:
        with lock:
            finished = True


def checkpoint():
    with lock:
        return published_gains.copy(), published_epoch


def log_state(data, displayed_epoch, error):
    ide.log("visualized_epoch", displayed_epoch)
    ide.log("elevator_height", float(data.qpos[0]))
    ide.log("target_height", target_at(data.time))
    ide.log("height_error", float(error))
    ide.log("elevator_velocity", float(data.qvel[0]))
    ide.log("left_motor_torque", float(data.ctrl[0]))
    ide.log("right_motor_torque", float(data.ctrl[1]))


def run_viewer():
    data = mujoco.MjData(model)
    integral = reset(data)
    gains, epoch = checkpoint()
    step_index = 0
    episode_loss = 0.0
    viewer_context = (
        __import__("pivot_viewer").launch_passive(model, data, str(MODEL_PATH))
        if os.environ.get("PIVOT_TELEMETRY_URL") else mujoco.viewer.launch_passive(model, data)
    )
    print("Starting epoch-0 visualization. Training starts AFTER its first full replay.")
    try:
        with viewer_context as viewer:
            while viewer.is_running():
                tick = time.perf_counter()
                if getattr(viewer, "reset_requested", False):
                    viewer.reset_requested = False
                    integral = reset(data)
                    step_index, episode_loss = 0, 0.0
                    gains, epoch = checkpoint()
                if getattr(viewer, "paused", False):
                    viewer.sync()
                    time.sleep(.025)
                    continue
                if step_index >= EPISODE_STEPS:
                    ide.log("episode_loss", float(episode_loss))
                    if not start_training.is_set():
                        print("Initial replay complete: starting optimizer.")
                        start_training.set()
                    integral = reset(data)
                    step_index, episode_loss = 0, 0.0
                    gains, epoch = checkpoint()
                    print(f"Replaying checkpoint epoch {epoch}")
                integral, delta, error = step(data, gains, integral)
                episode_loss += delta
                if step_index % 10 == 0:
                    log_state(data, epoch, error)
                step_index += 1
                viewer.sync()
                time.sleep(max(0.0, DT - (time.perf_counter() - tick)))
    finally:
        stop_training.set()
        start_training.set()


def run_embedded():
    # Pivot protocol: metadata + first frame, then arrays of frames on each request.
    sim_dir = os.environ["PIVOT_SIMULATOR_DIR"]
    sys.path.insert(0, sim_dir)
    from run import metadata, sample
    data = mujoco.MjData(model)
    integral = reset(data)
    gains, epoch = checkpoint()
    step_index, episode_loss = 0, 0.0
    if not sys.stdin.readline():
        return
    print(json.dumps({**metadata(model, MODEL_PATH.stem, mujoco),
                      "frames": [sample(model, data)]}), file=sys.__stdout__, flush=True)
    for line in sys.stdin:
        cmd = json.loads(line)
        if cmd.get("reset"):
            integral = reset(data)
            gains, epoch = checkpoint()
            step_index, episode_loss = 0, 0.0
            frames = [sample(model, data)]
        else:
            count, stride = int(cmd["count"]), int(cmd["stride"])
            if not (1 <= count <= 128 and 1 <= stride <= 4096 and count * stride <= 4096):
                raise ValueError("Invalid step batch")
            frames = []
            for _ in range(count):
                for _ in range(stride):
                    if step_index >= EPISODE_STEPS:
                        ide.log("episode_loss", float(episode_loss))
                        if not start_training.is_set():
                            start_training.set()
                        integral = reset(data)
                        gains, epoch = checkpoint()
                        step_index, episode_loss = 0, 0.0
                    integral, delta, error = step(data, gains, integral)
                    episode_loss += delta
                    step_index += 1
                frames.append(sample(model, data))
                if len(frames) % 2 == 0:
                    log_state(data, epoch, error)
        print(json.dumps(frames), file=sys.__stdout__, flush=True)
    stop_training.set()
    start_training.set()


if __name__ == "__main__":
    worker = threading.Thread(target=training_worker, daemon=True)
    worker.start()
    try:
        if EMBEDDED:
            run_embedded()
        else:
            run_viewer()
    finally:
        stop_training.set()
        start_training.set()
        worker.join(timeout=1)

