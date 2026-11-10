import time
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import mujoco
import mujoco.viewer
import ide_telem as ide

# ============================
# CONFIGURATION
# ============================

EMBEDDED = "--pivot-live" in sys.argv
if EMBEDDED:
    sys.stdout = sys.stderr  # Keep the controller's progress text off the frame protocol.
MODEL_PATH = Path(os.environ.get("PIVOT_MODEL_PATH", Path(__file__).with_name("pendulum.xml")))

# Desired joint angles (radians)
# [shoulder, elbow]
TARGET = np.array([-0.5, -0.5])
START = np.array([0.5, 0.5])

TRAIN_ITERATIONS = 200
EPISODE_DURATION = 3.0

# ============================
# MUJOCO SETUP
# ============================

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

assert model.nq == 2 and model.nv == 2 and model.nu == 2, (
    "This example expects the bundled 2-joint, 2-actuator pendulum."
)

dt = model.opt.timestep

# ============================
# PID CONTROLLER
# ============================

def pid_control(data, gains, integral):
    kp, ki, kd = gains

    position = data.qpos[:2]
    velocity = data.qvel[:2]

    error = TARGET - position

    integral += error * dt
    np.clip(integral, -2.0, 2.0, out=integral)

    torque = (
        kp * error
        + ki * integral
        - kd * velocity
    )

    # Respect MuJoCo actuator limits
    for j in range(2):
        if model.actuator_ctrllimited[j]:
            low, high = model.actuator_ctrlrange[j]
            torque[j] = np.clip(torque[j], low, high)

    return torque


# ============================
# SIMULATE ONE EPISODE
# ============================

def simulate(gains, visualize=False):
    data = mujoco.MjData(model)
    data.qpos[:2] = START
    mujoco.mj_forward(model, data)

    integral = np.zeros(2)
    total_loss = 0.0
    steps = int(EPISODE_DURATION / dt)

    def run_step():
        nonlocal total_loss

        torque = pid_control(data, gains, integral)
        data.ctrl[:2] = torque

        mujoco.mj_step(model, data)

        error = TARGET - data.qpos[:2]

        # Penalize position error, velocity, and effort
        loss = (
            np.sum(error ** 2)
            + 0.05 * np.sum(data.qvel[:2] ** 2)
            + 0.001 * np.sum(torque ** 2)
        )

        total_loss += loss * dt

        return error

    if visualize:
        WAIT_SECONDS = 4.0
        RUN_SECONDS = 3.0

        def reset_episode():
            mujoco.mj_resetData(model, data)
            data.qpos[:2] = START
            integral.fill(0.0)
            mujoco.mj_forward(model, data)

        reset_episode()

        print("Optimized PID gains:", gains)
        print("Target:", TARGET)
        print("Opening viewer...")

        if os.getenv("PIVOT_TELEMETRY_URL"):
            from pivot_viewer import launch_passive
            viewer_context = launch_passive(model, data, str(MODEL_PATH))
        else:
            viewer_context = mujoco.viewer.launch_passive(model, data)
        with viewer_context as viewer:
            phase = "waiting"
            phase_start = time.perf_counter()
            episode = 1
            step = 0

            print("Waiting 4 seconds...")

            while viewer.is_running():
                tick = time.perf_counter()
                if getattr(viewer, "reset_requested", False):
                    viewer.reset_requested = False
                    reset_episode()
                    phase = "waiting"
                    phase_start = tick
                if getattr(viewer, "paused", False):
                    viewer.sync()
                    time.sleep(0.025)
                    continue

                if phase == "waiting":
                    # Hold the pendulum at START
                    # without stepping physics.
                    if tick - phase_start >= WAIT_SECONDS:
                        phase = "running"
                        step = 0
                        print(f"Episode {episode}: PID active!")

                elif phase == "running":
                    if data.time >= RUN_SECONDS:
                        # Reset robot and PID integral
                        reset_episode()
                        episode += 1
                        phase = "waiting"
                        phase_start = tick
                        print("Reset! Waiting 4 seconds...")

                    else:
                        # Apply optimized PID
                        error = run_step()

                        if step % 10 == 0:
                            ide.log(
                                "shoulder_angle",
                                float(data.qpos[0])
                            )
                            ide.log(
                                "elbow_angle",
                                float(data.qpos[1])
                            )
                            ide.log(
                                "position_error",
                                float(np.linalg.norm(error))
                            )

                        step += 1

                viewer.sync()

                # Run simulation approximately in real time
                time.sleep(max(
                    0,
                    dt - (time.perf_counter() - tick)
                ))
    else:
        for _ in range(steps):
            run_step()

    return total_loss


# ============================
# TRAIN PID GAINS
# ============================

# Trainable gains: Kp, Ki, Kd
gains = torch.nn.Parameter(
    torch.tensor([8.0, 0.2, 1.5], dtype=torch.float64)
)

optimizer = torch.optim.Adam([gains], lr=0.3)

print("Training PID controller...")

for epoch in range(TRAIN_ITERATIONS):

    values = gains.detach().numpy().copy()
    loss = simulate(values)

    # Finite-difference gradient estimation
    # MuJoCo is not directly differentiable through PyTorch.
    gradient = np.zeros(3)
    epsilon = 0.05

    for j in range(3):
        plus = values.copy()
        minus = values.copy()

        plus[j] += epsilon
        minus[j] -= epsilon

        loss_plus = simulate(plus)
        loss_minus = simulate(minus)

        gradient[j] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

    # Give Adam the estimated gradient
    optimizer.zero_grad()
    gains.grad = torch.tensor(gradient, dtype=torch.float64)
    optimizer.step()

    # Keep gains positive and bounded
    with torch.no_grad():
        gains.clamp_(0.0, 50.0)

    kp, ki, kd = gains.detach().tolist()

    # Pivot telemetry
    ide.log("training_loss", float(loss))
    ide.log("Kp", kp)
    ide.log("Ki", ki)
    ide.log("Kd", kd)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {loss:.4f} | "
        f"Kp: {kp:.2f} "
        f"Ki: {ki:.2f} "
        f"Kd: {kd:.2f}"
    )

    time.sleep(0.05)


# ============================
# TEST TRAINED CONTROLLER
# ============================

final_gains = gains.detach().numpy().copy()

print("\nTraining complete!")
print("Optimized gains:", final_gains)
print("Target joint angles:", TARGET)

if EMBEDDED:
    sys.path.insert(0, os.environ["PIVOT_SIMULATOR_DIR"])
    from run import metadata, sample

    data = mujoco.MjData(model)
    integral = np.zeros(2)

    def reset_embedded():
        mujoco.mj_resetData(model, data)
        data.qpos[:2] = START
        integral.fill(0.0)
        mujoco.mj_forward(model, data)

    reset_embedded()
    # The first command acknowledges the trained controller and selected MJCF.
    if sys.stdin.readline():
        print(json.dumps({**metadata(model, MODEL_PATH.stem, mujoco), "frames": [sample(model, data)]}), file=sys.__stdout__, flush=True)
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("reset"):
                reset_embedded()
                frames = [sample(model, data)]
            else:
                count, stride = int(command["count"]), int(command["stride"])
                if not (1 <= count <= 128 and 1 <= stride <= 4096 and count * stride <= 4096):
                    raise ValueError("Invalid step batch")
                frames = []
                for _ in range(count):
                    for _ in range(stride):
                        data.ctrl[:2] = pid_control(data, final_gains, integral)
                        mujoco.mj_step(model, data)
                    frames.append(sample(model, data))
                    if len(frames) % 2 == 0:
                        ide.log("shoulder_angle", float(data.qpos[0]))
                        ide.log("elbow_angle", float(data.qpos[1]))
            print(json.dumps(frames), file=sys.__stdout__, flush=True)
else:
    print("\nLaunching MuJoCo viewer...")
    simulate(final_gains, visualize=True)

