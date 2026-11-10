
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
    sys.stdout = sys.stderr

MODEL_PATH = Path(
    os.environ.get(
        "PIVOT_MODEL_PATH",
        Path(__file__).with_name("cartpole.xml")
    )
)

TRAIN_ITERATIONS = 150
EPISODE_DURATION = 5.0

TARGET_CART = 0.0
TARGET_ANGLE = 0.0  # Upright = 0 radians

# Multiple initial disturbances for robust training
START_ANGLES = [-0.15, 0.15, 0.25]

# ============================
# MUJOCO SETUP
# ============================

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

assert model.nq == 2 and model.nv == 2 and model.nu == 1, (
    "Expected a 2-DOF cartpole with one cart actuator."
)

dt = model.opt.timestep

# Assumes cart slide joint first, pole hinge second
CART = 0
POLE = 1


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def reset_state(data, start_angle=0.15):
    mujoco.mj_resetData(model, data)

    data.qpos[CART] = 0.0
    data.qpos[POLE] = start_angle

    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)


# ============================
# STATE FEEDBACK CONTROLLER
# ============================

def cartpole_control(data, gains):
    k_theta, k_theta_d, k_x, k_x_d = gains

    x = data.qpos[CART]
    theta = wrap_angle(data.qpos[POLE])

    x_dot = data.qvel[CART]
    theta_dot = data.qvel[POLE]

    # Stabilize pole and return cart to origin
    force = (
        k_theta * theta
        + k_theta_d * theta_dot
        - k_x * (x - TARGET_CART)
        - k_x_d * x_dot
    )

    if model.actuator_ctrllimited[0]:
        low, high = model.actuator_ctrlrange[0]
        force = np.clip(force, low, high)

    return float(force)


# ============================
# SIMULATE ONE EPISODE
# ============================

def simulate(gains, start_angle=0.15, visualize=False):
    data = mujoco.MjData(model)
    reset_state(data, start_angle)

    total_loss = 0.0
    steps = int(EPISODE_DURATION / dt)

    def run_step():
        nonlocal total_loss

        force = cartpole_control(data, gains)
        data.ctrl[0] = force

        mujoco.mj_step(model, data)

        x = float(data.qpos[CART])
        theta = wrap_angle(float(data.qpos[POLE]))

        x_dot = float(data.qvel[CART])
        theta_dot = float(data.qvel[POLE])

        # Strongly prioritize keeping the pole upright
        loss = (
            20.0 * theta ** 2
            + 2.0 * x ** 2
            + 0.5 * theta_dot ** 2
            + 0.1 * x_dot ** 2
            + 0.0005 * force ** 2
        )

        total_loss += loss * dt

        return x, theta, force

    if visualize:
        WAIT_SECONDS = 2.0
        RUN_SECONDS = EPISODE_DURATION

        print("Optimized gains:", gains)
        print("Target: upright pole, centered cart")
        print("Opening viewer...")

        if os.getenv("PIVOT_TELEMETRY_URL"):
            from pivot_viewer import launch_passive
            viewer_context = launch_passive(
                model, data, str(MODEL_PATH)
            )
        else:
            viewer_context = mujoco.viewer.launch_passive(
                model, data
            )

        with viewer_context as viewer:
            phase = "waiting"
            phase_start = time.perf_counter()
            episode = 1
            step = 0

            while viewer.is_running():
                tick = time.perf_counter()

                if getattr(viewer, "reset_requested", False):
                    viewer.reset_requested = False
                    reset_state(data, start_angle)
                    phase = "waiting"
                    phase_start = tick

                if getattr(viewer, "paused", False):
                    viewer.sync()
                    time.sleep(0.025)
                    continue

                if phase == "waiting":
                    if tick - phase_start >= WAIT_SECONDS:
                        phase = "running"
                        step = 0
                        print(f"Episode {episode}: controller active!")

                elif phase == "running":
                    if data.time >= RUN_SECONDS:
                        reset_state(data, start_angle)
                        episode += 1
                        phase = "waiting"
                        phase_start = tick
                        print("Resetting cartpole...")

                    else:
                        x, theta, force = run_step()

                        if step % 10 == 0:
                            ide.log("cart_position", float(x))
                            ide.log("pole_angle", float(theta))
                            ide.log(
                                "pole_angle_degrees",
                                float(np.degrees(theta))
                            )
                            ide.log("cart_force", float(force))
                            ide.log(
                                "pole_angular_velocity",
                                float(data.qvel[POLE])
                            )

                        step += 1

                viewer.sync()

                time.sleep(max(
                    0,
                    dt - (time.perf_counter() - tick)
                ))

    else:
        for _ in range(steps):
            run_step()

            # Penalize failures aggressively
            theta = abs(wrap_angle(data.qpos[POLE]))
            x = abs(data.qpos[CART])

            if theta > 1.2 or x > 2.3:
                remaining = EPISODE_DURATION - data.time
                total_loss += 200.0 + 100.0 * max(remaining, 0)
                break

    return total_loss


# ============================
# EVALUATE CONTROLLER
# ============================

def evaluate(gains):
    losses = [
        simulate(gains, start_angle=angle)
        for angle in START_ANGLES
    ]

    return float(np.mean(losses))


# ============================
# TRAIN CONTROLLER GAINS
# ============================

# [Pole Kp, Pole Kd, Cart Kp, Cart Kd]
gains = torch.nn.Parameter(
    torch.tensor(
        [30.0, 8.0, 2.0, 4.0],
        dtype=torch.float64
    )
)

optimizer = torch.optim.Adam([gains], lr=0.3)

best_loss = float("inf")
best_gains = gains.detach().numpy().copy()

print("Training cartpole stabilization controller...")

for epoch in range(TRAIN_ITERATIONS):
    values = gains.detach().numpy().copy()
    loss = evaluate(values)

    gradient = np.zeros(4)
    epsilon = 0.05

    # Central finite differences
    for j in range(4):
        plus = values.copy()
        minus = values.copy()

        plus[j] += epsilon
        minus[j] -= epsilon

        loss_plus = evaluate(plus)
        loss_minus = evaluate(minus)

        gradient[j] = (
            loss_plus - loss_minus
        ) / (2.0 * epsilon)

    optimizer.zero_grad()

    gains.grad = torch.tensor(
        gradient,
        dtype=torch.float64
    )

    optimizer.step()

    # Keep gains within reasonable limits
    with torch.no_grad():
        gains.clamp_(0.0, 100.0)

    if loss < best_loss:
        best_loss = loss
        best_gains = values.copy()

    k_theta, k_theta_d, k_x, k_x_d = (
        gains.detach().tolist()
    )

    # Pivot telemetry
    ide.log("training_loss", float(loss))
    ide.log("best_loss", float(best_loss))

    ide.log("K_theta", k_theta)
    ide.log("K_theta_d", k_theta_d)
    ide.log("K_x", k_x)
    ide.log("K_x_d", k_x_d)

    print(
        f"Epoch {epoch + 1:03d} | "
        f"Loss: {loss:.4f} | "
        f"Best: {best_loss:.4f} | "
        f"Ktheta: {k_theta:.2f} | "
        f"Kdtheta: {k_theta_d:.2f} | "
        f"Kx: {k_x:.2f} | "
        f"Kdx: {k_x_d:.2f}"
    )

    time.sleep(0.02)


# ============================
# TEST TRAINED CONTROLLER
# ============================

final_gains = best_gains.copy()

print("\nTraining complete!")
print("Optimized gains:", final_gains)
print("Best training loss:", best_loss)

if EMBEDDED:
    sys.path.insert(
        0,
        os.environ["PIVOT_SIMULATOR_DIR"]
    )

    from run import metadata, sample

    data = mujoco.MjData(model)

    def reset_embedded():
        reset_state(data, 0.15)

    reset_embedded()

    # Initial metadata / frame response
    if sys.stdin.readline():
        print(
            json.dumps({
                **metadata(
                    model,
                    MODEL_PATH.stem,
                    mujoco
                ),
                "frames": [sample(model, data)]
            }),
            file=sys.__stdout__,
            flush=True
        )

        for line in sys.stdin:
            command = json.loads(line)

            if command.get("reset"):
                reset_embedded()
                frames = [sample(model, data)]

            else:
                count = int(command["count"])
                stride = int(command["stride"])

                if not (
                    1 <= count <= 128
                    and 1 <= stride <= 4096
                    and count * stride <= 4096
                ):
                    raise ValueError("Invalid step batch")

                frames = []

                for _ in range(count):
                    for _ in range(stride):
                        data.ctrl[0] = cartpole_control(
                            data,
                            final_gains
                        )

                        mujoco.mj_step(model, data)

                    frames.append(sample(model, data))

                    if len(frames) % 2 == 0:
                        ide.log(
                            "cart_position",
                            float(data.qpos[CART])
                        )
                        ide.log(
                            "pole_angle",
                            float(
                                wrap_angle(data.qpos[POLE])
                            )
                        )
                        ide.log(
                            "cart_force",
                            float(data.ctrl[0])
                        )

            print(
                json.dumps(frames),
                file=sys.__stdout__,
                flush=True
            )

else:
    print("\nLaunching MuJoCo viewer...")
    simulate(
        final_gains,
        start_angle=0.15,
        visualize=True
    )

