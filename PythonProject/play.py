import gymnasium as gym
import flappy_bird_gymnasium
import torch
import time

from model.DQN import DQN
from memory.frame_stack import FrameStack
from utils.preprocess import preprocess_frame

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_human = gym.make("FlappyBird-v0", render_mode="human")

env_rgb = gym.make("FlappyBird-v0", render_mode="rgb_array")

policy_net = DQN().to(DEVICE)
policy_net.load_state_dict(torch.load("dqn_flappy.pth", map_location=DEVICE))
policy_net.eval()

env_human.reset()
env_rgb.reset()

frame = preprocess_frame(env_rgb.render())
stacker = FrameStack(4)
state = stacker.reset(frame)

done = False
total_reward = 0

while not done:
    with torch.no_grad():
        state_t = torch.tensor(state).unsqueeze(0).to(DEVICE)
        action = policy_net(state_t).argmax(1).item()

    _, reward, terminated, truncated, _ = env_human.step(action)
    env_rgb.step(action)

    done = terminated or truncated

    next_frame = preprocess_frame(env_rgb.render())
    state = stacker.step(next_frame)

    total_reward += reward
    time.sleep(0.02)

print("Final reward:", total_reward)

env_human.close()
env_rgb.close()
