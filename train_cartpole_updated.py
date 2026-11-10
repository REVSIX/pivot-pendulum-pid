"""Live-training cart-pole state-feedback controller for MuJoCo / Pivot.

Run standalone:
    python train_cartpole.py

Run embedded through Pivot with --pivot-live and PIVOT_SIMULATOR_DIR.
The viewer starts with the untrained controller (epoch 0), and training
publishes new gains every UPDATE_EVERY epochs without blocking simulation.
The viewer adopts the latest gains at the beginning of each new episode.

Expects a 2-DoF cartpole with a cart slide joint, pole hinge joint, and one
motor on the cart. Prefer names cart_slide and pole_hinge; otherwise use qpos
indices 0 and 1, respectively.
"""

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np
import torch

import ide_telem as ide

EMBEDDED = "--pivot-live" in sys.argv
if EMBEDDED:
    # The embedded protocol owns stdout; progress prints go to stderr.
    sys.stdout = sys.stderr

MODEL_PATH = Path(
    os.environ.get("PIVOT_MODEL_PATH", Path(__file__).with_name("cartpole.xml"))
)
TRAIN_ITERATIONS = 150
UPDATE_EVERY = 10
EPISODE_DURATION = 5.0
VISUAL_START_ANGLE = 0.25
TRAIN_START_ANGLES = (-0.15, 0.15, 0.25)
FAIL_ANGLE = 1.2
FAIL_CART_POSITION = 2.3
START_GAINS = np.array([30.0, 8.0, 2.0, 4.0], dtype=np.float64)
LEARNING_RATE = 0.3
FD_EPSILON = 0.05

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
if not (model.nq == 2 and model.nv == 2 and model.nu == 1):
    raise ValueError("Expected a 2-DoF cartpole with one cart actuator.")

# Joint ordering in the bundled cartpole.xml is cart slide, then pole hinge.
# Verify, instead of silently interpreting the coordinates in the wrong order.
cart_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cart_slide")
pole_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pole_hinge")
if cart_joint < 0 or pole_joint < 0:
    cart_qpos, pole_qpos = 0, 1
    cart_qvel, pole_qvel = 0, 1
else:
    cart_qpos = int(model.jnt_qposadr[cart_joint])
    pole_qpos = int(model.jnt_qposadr[pole_joint])
    cart_qvel = int(model.jnt_dofadr[cart_joint])
    pole_qvel = int(model.jnt_dofadr[pole_joint])

DT = float(model.opt.timestep)
EPISODE_STEPS = max(1, round(EPISODE_DURATION / DT))


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def reset_state(data: mujoco.MjData, start_angle: float = VISUAL_START_ANGLE) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[cart_qpos] = 0.0
    data.qpos[pole_qpos] = start_angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def cartpole_control(data: mujoco.MjData, gains: np.ndarray) -> float:
    kp_theta, kd_theta, kp_cart, kd_cart = gains
    x = float(data.qpos[cart_qpos])
    theta = wrap_angle(float(data.qpos[pole_qpos]))
    x_dot = float(data.qvel[cart_qvel])
    theta_dot = float(data.qvel[pole_qvel])
    force = (kp_theta * theta + kd_theta * theta_dot
             - kp_cart * x - kd_cart * x_dot)
    if model.actuator_ctrllimited[0]:
        low, high = model.actuator_ctrlrange[0]
        force = np.clip(force, low, high)
    return float(force)


def step_controller(data: mujoco.MjData, gains: np.ndarray) -> float:
    force = cartpole_control(data, gains)
    data.ctrl[0] = force
    mujoco.mj_step(model, data)
    return force


def state_loss(data: mujoco.MjData, force: float) -> float:
    theta = wrap_angle(float(data.qpos[pole_qpos]))
    x = float(data.qpos[cart_qpos])
    theta_dot = float(data.qvel[pole_qvel])
    x_dot = float(data.qvel[cart_qvel])
    return (20.0 * theta**2 + 2.0 * x**2 + 0.5 * theta_dot**2
            + 0.1 * x_dot**2 + 0.0005 * force**2) * DT


def failed(data: mujoco.MjData) -> bool:
    return (abs(wrap_angle(float(data.qpos[pole_qpos]))) > FAIL_ANGLE
            or abs(float(data.qpos[cart_qpos])) > FAIL_CART_POSITION)


def simulate_training(gains: np.ndarray, start_angle: float) -> float:
    # Training uses its OWN MjData; never shares viewer/embedded simulation state.
    data = mujoco.MjData(model)
    reset_state(data, start_angle)
    loss = 0.0
    for step in range(EPISODE_STEPS):
        force = step_controller(data, gains)
        loss += state_loss(data, force)
        if failed(data):
            remaining = (EPISODE_STEPS - step - 1) * DT
            loss += 200.0 + 100.0 * remaining
            break
    return float(loss)


def evaluate(gains: np.ndarray) -> float:
    return float(np.mean([
        simulate_training(gains, angle) for angle in TRAIN_START_ANGLES
    ]))


class SharedTrainingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.display_gains = START_GAINS.copy()
        self.display_epoch = 0
        self.completed_epoch = 0
        self.current_loss = None
        self.best_gains = START_GAINS.copy()
        self.best_loss = float("inf")
        self.finished = False
        self.error = None

    def snapshot(self):
        with self.lock:
            return (self.display_gains.copy(), self.display_epoch,
                    self.completed_epoch, self.current_loss,
                    self.finished, self.error)


shared = SharedTrainingState()


def train_background() -> None:
    """Training runs off the viewer thread and publishes checkpoints atomically."""
    try:
        torch.set_num_threads(1)
        gains = torch.nn.Parameter(torch.tensor(START_GAINS, dtype=torch.float64))
        optimizer = torch.optim.Adam([gains], lr=LEARNING_RATE)
        print("Training started. Viewer is replaying epoch 0 controller.")

        for epoch_index in range(TRAIN_ITERATIONS):
            current = gains.detach().numpy().copy()
            loss = evaluate(current)
            gradient = np.zeros(4, dtype=np.float64)
            for j in range(4):
                plus, minus = current.copy(), current.copy()
                plus[j] += FD_EPSILON
                minus[j] -= FD_EPSILON
                gradient[j] = ((evaluate(plus) - evaluate(minus))
                               / (2.0 * FD_EPSILON))

            optimizer.zero_grad()
            gains.grad = torch.tensor(gradient, dtype=torch.float64)
            optimizer.step()
            with torch.no_grad():
                gains.clamp_(0.0, 100.0)

            completed = epoch_index + 1
            optimized = gains.detach().numpy().copy()
            # Report the loss of the snapshot actually being published.
            checkpoint_due = (completed % UPDATE_EVERY == 0
                              or completed == TRAIN_ITERATIONS)
            checkpoint_loss = evaluate(optimized) if checkpoint_due else None

            with shared.lock:
                if loss < shared.best_loss:
                    shared.best_loss = loss
                    shared.best_gains = current.copy()
                if checkpoint_due and checkpoint_loss < shared.best_loss:
                    shared.best_loss = checkpoint_loss
                    shared.best_gains = optimized.copy()
                shared.completed_epoch = completed
                shared.current_loss = float(loss)
                if checkpoint_due:
                    shared.display_gains = optimized.copy()
                    shared.display_epoch = completed
                best_loss = shared.best_loss

            ide.log("training_loss", float(loss))
            ide.log("best_loss", float(best_loss))
            ide.log("training_epoch", float(completed))
            ide.log("K_theta", float(optimized[0]))
            ide.log("K_theta_d", float(optimized[1]))
            ide.log("K_x", float(optimized[2]))
            ide.log("K_x_d", float(optimized[3]))

            print(f"Epoch {completed:03d}/{TRAIN_ITERATIONS} | "
                  f"loss={loss:.4f} | gains={np.round(optimized, 2)}"
                  + (" | PUBLISHED" if checkpoint_due else ""))

        with shared.lock:
            shared.finished = True
        print("Training completed; viewer continues replaying final checkpoint.")
    except Exception:
        message = traceback.format_exc()
        with shared.lock:
            shared.error = message
            shared.finished = True
        print("Training error:\n" + message, file=sys.stderr)


def log_visual_telemetry(data: mujoco.MjData, epoch: int, loss: float) -> None:
    ide.log("visualized_epoch", float(epoch))
    ide.log("pole_angle", wrap_angle(float(data.qpos[pole_qpos])))
    ide.log("cart_position", float(data.qpos[cart_qpos]))
    ide.log("cart_force", float(data.ctrl[0]))
    ide.log("pole_angular_velocity", float(data.qvel[pole_qvel]))
    ide.log("episode_loss", float(loss))


def run_standalone_viewer() -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    active_gains, active_epoch, *_ = shared.snapshot()
    reset_state(data)
    episode_steps = 0
    episode_loss = 0.0
    episode_number = 1

    if os.getenv("PIVOT_TELEMETRY_URL"):
        from pivot_viewer import launch_passive
        viewer_context = launch_passive(model, data, str(MODEL_PATH))
    else:
        viewer_context = mujoco.viewer.launch_passive(model, data)

    # The viewer is opened BEFORE training begins, so epoch 0 is visible.
    with viewer_context as viewer:
        print("Viewer ready: showing INITIAL (epoch 0) controller.")
        thread = threading.Thread(target=train_background, daemon=True)
        thread.start()

        while viewer.is_running():
            tick = time.perf_counter()
            if getattr(viewer, "paused", False):
                viewer.sync()
                time.sleep(0.025)
                continue

            if getattr(viewer, "reset_requested", False):
                viewer.reset_requested = False
                episode_steps = EPISODE_STEPS  # Force a fresh episode.

            # Freeze the gains for a whole episode; adopt the most recent
            # published snapshot only at the NEXT episode boundary.
            if episode_steps >= EPISODE_STEPS or failed(data):
                print(f"Replay {episode_number}: displayed epoch {active_epoch}, "
                      f"episode loss {episode_loss:.3f}")
                new_gains, new_epoch, *_ = shared.snapshot()
                if new_epoch != active_epoch:
                    print(f"Controller updated: epoch {active_epoch} -> {new_epoch}")
                active_gains, active_epoch = new_gains, new_epoch
                reset_state(data)
                episode_number += 1
                episode_steps = 0
                episode_loss = 0.0

            force = step_controller(data, active_gains)
            episode_loss += state_loss(data, force)
            episode_steps += 1

            if episode_steps % 10 == 0:
                log_visual_telemetry(data, active_epoch, episode_loss)
            viewer.sync()
            time.sleep(max(0.0, DT - (time.perf_counter() - tick)))


def run_embedded() -> None:
    # Pivot owns a request/response protocol on stdout. Do not emit any
    # progress text there, including messages from the training thread.
    sys.path.insert(0, os.environ["PIVOT_SIMULATOR_DIR"])
    from run import metadata, sample

    data = mujoco.MjData(model)
    active_gains, active_epoch, *_ = shared.snapshot()
    episode_steps = 0
    episode_loss = 0.0
    reset_state(data)

    # Publish epoch-0 frame before training. Start training after the initial
    # handshake so the first visualization is always pre-training.
    if not sys.stdin.readline():
        return
    print(json.dumps({
        **metadata(model, MODEL_PATH.stem, mujoco),
        "frames": [sample(model, data)],
    }), file=sys.__stdout__, flush=True)

    thread = threading.Thread(target=train_background, daemon=True)
    thread.start()

    for line in sys.stdin:
        command = json.loads(line)
        if command.get("reset"):
            reset_state(data)
            episode_steps = 0
            episode_loss = 0.0
            active_gains, active_epoch, *_ = shared.snapshot()
            frames = [sample(model, data)]
        else:
            count, stride = int(command["count"]), int(command["stride"])
            if not (1 <= count <= 128 and 1 <= stride <= 4096
                    and count * stride <= 4096):
                raise ValueError("Invalid step batch")
            frames = []
            for _ in range(count):
                for _ in range(stride):
                    if episode_steps >= EPISODE_STEPS or failed(data):
                        active_gains, active_epoch, *_ = shared.snapshot()
                        reset_state(data)
                        episode_steps = 0
                        episode_loss = 0.0
                    force = step_controller(data, active_gains)
                    episode_loss += state_loss(data, force)
                    episode_steps += 1
                    if episode_steps % 10 == 0:
                        log_visual_telemetry(data, active_epoch, episode_loss)
                frames.append(sample(model, data))
        print(json.dumps(frames), file=sys.__stdout__, flush=True)


if __name__ == "__main__":
    if EMBEDDED:
        run_embedded()
    else:
        run_standalone_viewer()

