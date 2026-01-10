import gymnasium as gym
import flappy_bird_gymnasium

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from model.DQN import DQN
from memory.replay_buffer import ReplayBuffer
from memory.frame_stack import FrameStack
from utils.preprocess import preprocess_frame

# =========================
# Hyperparameters
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
LR = 2.5e-4

BATCH_SIZE = 32
REPLAY_CAPACITY = 100_000

LEARNING_STARTS = 10_000
TRAIN_EVERY = 4
TARGET_UPDATE = 1000
MAX_GRAD_NORM = 10.0

NUM_EPISODES = 1500

EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 15_000   # faster learning (as requested)

ACTION_INTERVAL = 5  # decision every 5 frames

# =========================
# Environment
# =========================
env = gym.make(
    "FlappyBird-v0",
    render_mode="rgb_array",
    disable_env_checker=True,
)

# =========================
# Networks
# =========================
policy_net = DQN().to(DEVICE)
target_net = DQN().to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.SmoothL1Loss()

memory = ReplayBuffer(REPLAY_CAPACITY)
steps_done = 0

# =========================
# Action selection
# =========================
def select_action(state):
    global steps_done

    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
    steps_done += 1

    if random.random() < eps:
        return random.randint(0, 1)

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return policy_net(state_t).argmax(1).item()

# =========================
# Optimization (Double DQN)
# =========================
def optimize_model():
    if len(memory) < BATCH_SIZE or steps_done < LEARNING_STARTS:
        return

    states, actions, rewards, next_states, dones = zip(*memory.sample(BATCH_SIZE))

    states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        next_q = target_net(next_states).gather(1, next_actions)
        target_q = rewards + GAMMA * next_q * (1.0 - dones)

    loss = loss_fn(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), MAX_GRAD_NORM)
    optimizer.step()

# =========================
# Training Loop
# =========================
for episode in range(NUM_EPISODES):
    env.reset()

    frame_counter = 0
    prev_pipe_x = None

    frame = preprocess_frame(env.render())
    stacker = FrameStack(4)
    state = stacker.reset(frame)

    done = False
    total_reward = 0.0

    current_action = 0
    decision_state = state
    decision_reward = 0.0

    while not done:
        is_decision_frame = (frame_counter % ACTION_INTERVAL == 0)

        if is_decision_frame:
            current_action = select_action(state)
            decision_state = state
            decision_reward = 0.0
            action = current_action
        else:
            action = 0  # NO-OP

        frame_counter += 1

        # Step environment
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_frame = preprocess_frame(env.render())
        next_state = stacker.step(next_frame)

        # =========================
        # Reward (FRAME-LEVEL)
        # =========================
        reward = 0.0

        # alive
        if not terminated:
            reward += 0.1

        # death
        if terminated:
            reward -= 1.0

        # ceiling penalty
        bird_pixels = next_frame.sum(axis=1)
        bird_y = np.argmax(bird_pixels)
        if bird_y < 3:
            reward -= 0.5

        # pipe passing
        pipe_columns = next_frame.sum(axis=0)
        pipe_positions = np.where(pipe_columns > 0)[0]
        if len(pipe_positions) > 0:
            pipe_x = pipe_positions[0]
            if prev_pipe_x is not None and pipe_x < prev_pipe_x:
                reward += 1.0
            prev_pipe_x = pipe_x

        # =========================
        # Accumulate decision reward
        # =========================
        decision_reward += reward
        total_reward += reward

        # =========================
        # Store transition (decision only)
        # =========================
        if is_decision_frame:
            memory.push(
                decision_state,
                current_action,
                decision_reward,
                next_state,
                done
            )

        state = next_state

        # Train & sync
        if steps_done % TRAIN_EVERY == 0:
            optimize_model()

        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print(
        f"Episode {episode:4d} | "
        f"Reward {total_reward:7.2f} | "
        f"Epsilon {EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY):.3f} | "
        f"Buffer {len(memory)}"
    )

    if episode % 25 == 0:
        torch.save(policy_net.state_dict(), "dqn_flappy.pth")

env.close()
