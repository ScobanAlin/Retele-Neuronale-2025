import gymnasium as gym
import flappy_bird_gymnasium

def make_env(render=False):
    return gym.make(
        "FlappyBird-v0",
        render_mode="rgb_array" if not render else "human"
    )

env = make_env(render=True)
env.reset()
while True:
    env.step(0)