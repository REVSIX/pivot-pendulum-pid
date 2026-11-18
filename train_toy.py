
import os
import sys
import json
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
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
        Path(__file__).with_name("toy.xml")
    )
)

TRAIN_EPISODES = 200
EPISODE_DURATION = 3.0
UPDATE_EVERY = 10

START_HEIGHT = 0.3

ALPHA = 0.2
GAMMA = 0.95
INITIAL_EPSILON = 0.3
MIN_EPSILON = 0.03

# ============================
# MUJOCO SETUP
# ============================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

assert model.nq == 1 and model.nv == 1 and model.nu == 1, (
    "Expected one vertical slide joint and one thruster."
)

dt = float(model.opt.timestep)
EPISODE_STEPS = round(EPISODE_DURATION / dt)

# ============================
# Q-LEARNING SETUP
# ============================

height_bins = np.linspace(0.1, 3.0, 15)
vel_bins = np.linspace(-10.0, 10.0, 15)

ACTIONS = np.array(
    [-20.0, 0.0, 20.0],
    dtype=np.float64
)

Q_SHAPE = (
    len(height_bins) + 1,
    len(vel_bins) + 1,
    len(ACTIONS)
)

q_table = np.zeros(Q_SHAPE, dtype=np.float64)

rng = np.random.default_rng(42)


def get_state_idx(height, velocity):
    h_idx = int(np.digitize(height, height_bins))
    v_idx = int(np.digitize(velocity, vel_bins))
    return h_idx, v_idx


def select_action(q, state, epsilon=0.0):
    if rng.random() < epsilon:
        return int(rng.integers(len(ACTIONS)))

    values = q[state]

    # For untrained states with equal Q-values,
    # prefer coasting rather than full down thrust.
    if np.all(values == values[0]):
        return 1

    return int(np.argmax(values))


def reset_state(data):
    mujoco.mj_resetData(model, data)

    data.qpos[0] = START_HEIGHT
    data.qvel[0] = 0.0

    mujoco.mj_forward(model, data)


# ============================
# SHARED TRAINING STATE
# ============================

policy_lock = threading.Lock()
training_start = threading.Event()
stop_training = threading.Event()

# Snapshot queue: preserve every 10-episode
# milestone, even if training runs ahead.
checkpoints = deque()

display_q = q_table.copy()
display_episode = 0

training_episode = 0
training_finished = False
best_reward = float("-inf")


def publish_checkpoint(episode):
    with policy_lock:
        checkpoints.append(
            (episode, q_table.copy())
        )


def next_display_checkpoint():
    global display_q, display_episode

    with policy_lock:
        if checkpoints:
            display_episode, display_q = (
                checkpoints.popleft()
            )


# ============================
# TRAIN ONE EPISODE
# ============================

def train_episode(data, epsilon):
    reset_state(data)

    total_reward = 0.0
    maximum_height = float(data.qpos[0])

    for step in range(EPISODE_STEPS):
        if stop_training.is_set():
            break

        state = get_state_idx(
            data.qpos[0],
            data.qvel[0]
        )

        action_idx = select_action(
            q_table,
            state,
            epsilon
        )

        thrust = ACTIONS[action_idx]
        data.ctrl[0] = thrust

        mujoco.mj_step(model, data)

        height = float(data.qpos[0])
        velocity = float(data.qvel[0])

        next_state = get_state_idx(
            height,
            velocity
        )

        # Reward altitude; mildly penalize effort.
        reward = (
            2.0 * height
            - 0.001 * thrust ** 2
        )

        total_reward += reward
        maximum_height = max(
            maximum_height,
            height
        )

        best_future_q = np.max(
            q_table[next_state]
        )

        old_q = q_table[
            state[0],
            state[1],
            action_idx
        ]

        target_q = (
            reward
            + GAMMA * best_future_q
        )

        q_table[
            state[0],
            state[1],
            action_idx
        ] = (
            old_q
            + ALPHA * (target_q - old_q)
        )

    avg_reward = total_reward / EPISODE_STEPS

    return avg_reward, maximum_height


# ============================
# BACKGROUND TRAINING
# ============================

def train_background():
    global training_episode
    global training_finished
    global best_reward

    # Crucial: do not start training until
    # the initial untrained replay finishes.
    training_start.wait()

    if stop_training.is_set():
        return

    data = mujoco.MjData(model)

    print("\nStarting Q-learning...")

    for episode in range(1, TRAIN_EPISODES + 1):
        if stop_training.is_set():
            return

        progress = (episode - 1) / TRAIN_EPISODES

        epsilon = max(
            MIN_EPSILON,
            INITIAL_EPSILON * (1.0 - progress)
        )

        reward, max_height = train_episode(
            data,
            epsilon
        )

        training_episode = episode
        best_reward = max(best_reward, reward)

        ide.log("training_reward", float(reward))
        ide.log("best_reward", float(best_reward))
        ide.log("exploration_epsilon", float(epsilon))
        ide.log("training_episode", float(episode))
        ide.log("training_max_height", float(max_height))

        if episode % UPDATE_EVERY == 0:
            publish_checkpoint(episode)

            print(
                f"Episode {episode:03d}/{TRAIN_EPISODES} | "
                f"Reward: {reward:.3f} | "
                f"Max height: {max_height:.2f}m | "
                f"Epsilon: {epsilon:.3f}"
            )

    with policy_lock:
        training_finished = True

    print("\nQ-learning complete!")
    print("All policy checkpoints published.")


# ============================
# LIVE VISUALIZATION
# ============================

live_data = mujoco.MjData(model)

replay_number = 0
replay_step = 0
replay_reward = 0.0

reset_state(live_data)


def reset_replay():
    global replay_step
    global replay_reward

    reset_state(live_data)
    replay_step = 0
    replay_reward = 0.0


def finish_replay():
    global replay_number

    replay_number += 1

    avg_reward = replay_reward / EPISODE_STEPS

    ide.log(
        "visualization_reward",
        float(avg_reward)
    )

    print(
        f"Replay {replay_number} | "
        f"Policy episode: {display_episode} | "
        f"Reward: {avg_reward:.3f}"
    )

    # The very first replay must show the
    # untrained controller before learning.
    if replay_number == 1:
        print("Initial replay complete. Training enabled.")
        training_start.set()

    next_display_checkpoint()
    reset_replay()


def visualization_step():
    global replay_step
    global replay_reward

    height = float(live_data.qpos[0])
    velocity = float(live_data.qvel[0])

    state = get_state_idx(height, velocity)

    # Visualization is always greedy:
    # no exploration during demonstrations.
    action_idx = select_action(
        display_q,
        state,
        epsilon=0.0
    )

    thrust = float(ACTIONS[action_idx])
    live_data.ctrl[0] = thrust

    mujoco.mj_step(model, live_data)

    height = float(live_data.qpos[0])
    velocity = float(live_data.qvel[0])

    reward = (
        2.0 * height
        - 0.001 * thrust ** 2
    )

    replay_reward += reward
    replay_step += 1

    # Live telemetry
    if replay_step % 10 == 0:
        ide.log("toy_height", height)
        ide.log("toy_velocity", velocity)
        ide.log("toy_thrust", thrust)
        ide.log(
            "visualized_episode",
            float(display_episode)
        )
        ide.log(
            "target_reward_rate",
            float(reward)
        )

    if replay_step >= EPISODE_STEPS:
        finish_replay()


# ============================
# START TRAINING WORKER
# ============================

worker = threading.Thread(
    target=train_background,
    daemon=True
)

worker.start()

print("Bouncy Toy: Q-Learning")
print("Initial policy: untrained")
print(f"Replay duration: {EPISODE_DURATION}s")
print(f"Policy updates: every {UPDATE_EVERY} episodes")


# ============================
# PIVOT EMBEDDED MODE
# ============================

if EMBEDDED:
    sys.path.insert(
        0,
        os.environ["PIVOT_SIMULATOR_DIR"]
    )

    from run import metadata, sample

    try:
        if sys.stdin.readline():
            print(
                json.dumps({
                    **metadata(
                        model,
                        MODEL_PATH.stem,
                        mujoco
                    ),
                    "frames": [
                        sample(model, live_data)
                    ]
                }),
                file=sys.__stdout__,
                flush=True
            )

            for line in sys.stdin:
                command = json.loads(line)

                if command.get("reset"):
                    reset_replay()
                    frames = [
                        sample(model, live_data)
                    ]

                else:
                    count = int(command["count"])
                    stride = int(command["stride"])

                    if not (
                        1 <= count <= 128
                        and 1 <= stride <= 4096
                        and count * stride <= 4096
                    ):
                        raise ValueError(
                            "Invalid step batch"
                        )

                    frames = []

                    for _ in range(count):
                        for _ in range(stride):
                            visualization_step()

                        frames.append(
                            sample(model, live_data)
                        )

                print(
                    json.dumps(frames),
                    file=sys.__stdout__,
                    flush=True
                )

    finally:
        stop_training.set()
        training_start.set()


# ============================
# STANDALONE MUJOCO VIEWER
# ============================

else:
    print("\nOpening live visualization...")

    if os.getenv("PIVOT_TELEMETRY_URL"):
        from pivot_viewer import launch_passive

        viewer_context = launch_passive(
            model,
            live_data,
            str(MODEL_PATH)
        )

    else:
        viewer_context = mujoco.viewer.launch_passive(
            model,
            live_data
        )

    try:
        with viewer_context as viewer:
            while viewer.is_running():
                tick = time.perf_counter()

                if getattr(viewer, "reset_requested", False):
                    viewer.reset_requested = False
                    reset_replay()

                if getattr(viewer, "paused", False):
                    viewer.sync()
                    time.sleep(0.025)
                    continue

                visualization_step()
                viewer.sync()

                time.sleep(
                    max(
                        0.0,
                        dt - (
                            time.perf_counter() - tick
                        )
                    )
                )

    finally:
        stop_training.set()
        training_start.set()
        worker.join(timeout=2.0)

